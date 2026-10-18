"""
Panel figure of a turbulence ensemble (one frame from each of the 20 simulations)
plus the frame-to-frame correlation matrix, to check the frames are decorrelated.

Usage: python3 plot_swe_ensemble.py          # SWE ensemble, frame 0 of each sim
       python3 plot_swe_ensemble.py alpha    # alpha=2 ensemble, output 90 of each seed
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) > 1 and sys.argv[1] == "alpha":
    data_path = "./JWS_turb_data/alpha-turb/alpha2.0_vorticity_20_independent_runs_output0090.npz"
    out_png = "./figures/alpha_ensemble_frames.png"
    vmin, vmax = -0.06, 0.06
    with np.load(data_path) as data:
        frames = data["vorticity"]
        t = data["time"].mean()
    title = rf"$\alpha$ = 2 turbulence ensemble, output 90 of 20 independent runs (mean t = {t:.1f})"
else:
    data_path = "./JWS_turb_data/swe-turb/SWE_vorticity_20_sim_ensemble_10_times.npz"
    frame_idx = 0
    out_png = "./figures/swe_ensemble_frames.png"
    vmin, vmax = -0.15, 0.15
    with np.load(data_path) as data:
        frames = data["vorticity"][:, frame_idx, :, :]
        t = data["time"][frame_idx]
    title = f"SWE turbulence ensemble, frame {frame_idx} (t = {t:.1f})"
num_sims = frames.shape[0]

# Pearson correlation between every pair of simulations
flat = frames.reshape(num_sims, -1)
corr = np.corrcoef(flat)
off_diag = np.abs(corr[~np.eye(num_sims, dtype=bool)])

fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(4, 8, width_ratios=[1, 1, 1, 1, 1, 0.15, 0.3, 3.2], wspace=0.05, hspace=0.12)

for i in range(num_sims):
    ax = fig.add_subplot(gs[i // 5, i % 5])
    im = ax.imshow(frames[i], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(f"sim {i}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
cax = fig.add_subplot(gs[:, 5])
fig.colorbar(im, cax=cax, label=r"Vorticity $\zeta$")

ax = fig.add_subplot(gs[1:3, 7])
imc = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(0, num_sims, 2))
ax.set_yticks(range(0, num_sims, 2))
ax.set_xlabel("sim")
ax.set_ylabel("sim")
ax.set_title(f"Correlation between sims\nmean |r| = {off_diag.mean():.3f}, max |r| = {off_diag.max():.3f}",
             fontsize=10)
fig.colorbar(imc, ax=ax, fraction=0.046, pad=0.04, label="r")

fig.suptitle(title, fontsize=13)
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved {out_png}")
print(f"Off-diagonal |r|: mean {off_diag.mean():.3f}, max {off_diag.max():.3f}")
