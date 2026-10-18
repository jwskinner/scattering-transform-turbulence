"""
S1/S2-only scattering transform synthesis of the SWE turbulence ensemble
(cell 16 of Compare_Powerspectrum_Bispectrum_ST.ipynb), using the same
scattering2.synthesis() routine as the ICE_ST synthesis notebook.

Usage: python3 synthesize_st_s1s2.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/jskinner1")
from scattering2 import synthesis

# ------- Settings (ICE_ST synthesis settings) ------
data_path = "./JWS_turb_data/swe-turb/SWE_vorticity_20_sim_ensemble_10_times.npz"
frame_idx = 0
J, L = 8, 8
loss_type = "normalized_mse"
learnable_param_list = [(200, 1.0)]
out_png = "./figures/st_s1s2_synthesis_swe.png"
out_npz = "./figures/st_s1s2_synthesis_swe.npz"
vmin, vmax = -0.1, 0.1

# ------- Load target ensemble (20 sims, one frame) ------
with np.load(data_path) as data:
    image = data["vorticity"][:, frame_idx, :, :].astype(np.float32)
num_fields, M, N = image.shape
print(f"Target ensemble: {image.shape}")

# z-score like the ICE fields so Adam's lr has the same meaning; S1 and S2 scale
# linearly with amplitude, so this does not change which field is synthesized
mu, sigma = image.mean(), image.std()
image_z = (image - mu) / sigma

# ------- Synthesis: S1 and S2 only, no histogram, no phase harmonics ------
synth_z, loss_history = synthesis(
    image_z,
    M=M, N=N, J=J, L=L,
    j_min=1, j_max=None,
    loss_type=loss_type,
    learnable_param_list=learnable_param_list,
    histogram_loss=False,
    phase_harmonics=False,
    plot_progress=False,
    plot_loss=False,
)
synth = synth_z[0] * sigma + mu
target = image[0]
np.savez(out_npz, synth=synth, target=target, total_loss=np.array(loss_history["total"]))


# ------- Figure (same panels as plot_synthesis in the notebook) ------
def radial_power_spectrum(img, bins=20):
    """Same binning as ST.get_power_spectrum."""
    M, N = img.shape
    modulus = np.fft.fftshift(np.abs(np.fft.fft2(img)))
    Y, X = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    R = np.sqrt((X - M / 2) ** 2 + (Y - N / 2) ** 2)
    edges = np.concatenate(([0], np.logspace(0.0, np.log10(1.4 * M / 2), bins)))
    ps = np.array([modulus[(R >= lo) & (R < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    return ps, edges


fig, axes = plt.subplots(1, 5, figsize=(18, 3))
axes[0].imshow(target, vmin=vmin, vmax=vmax, cmap="RdBu")
axes[0].set_title("Target Image")
axes[1].imshow(synth, vmin=vmin, vmax=vmax, cmap="RdBu")
axes[1].set_title("Synthesized Image (S1, S2)")
for ax in axes[:2]:
    ax.set_xticks([])
    ax.set_yticks([])

for img, label in [(target, "target"), (synth, "synthesized")]:
    y, x = np.histogram(img, 100, (vmin, vmax))
    axes[2].plot((x[1:] + x[:-1]) / 2, y, label=label)
axes[2].set_yscale("log")
axes[2].set_title("Pixel Intensity Histogram")
axes[2].legend()

ps_t, edges = radial_power_spectrum(target)
ps_s, _ = radial_power_spectrum(synth)
k = (edges[:-1] + edges[1:]) / 2
axes[3].loglog(k, ps_t, label="target")
axes[3].loglog(k, ps_s, label="synthesized", linestyle="--")
axes[3].set_ylim(bottom=1e-5)
axes[3].set_xlabel("k")
axes[3].set_title("Power Spectrum")
axes[3].legend()

for key, vals in loss_history.items():
    if len(vals) > 0:
        axes[4].plot(vals, label=key, color="black" if key == "total" else None,
                     linestyle="--" if key == "total" else "-")
axes[4].set_yscale("log")
axes[4].set_xlabel("Step")
axes[4].set_title("Loss")
axes[4].legend(frameon=False)

plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved {out_png}")
print(f"Target STD: {target.std():.5f} | Synth STD: {synth.std():.5f}")
