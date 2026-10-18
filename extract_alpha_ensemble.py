"""
Build an ensemble of independent alpha-turbulence vorticity snapshots: one frame
(the same output index) from each of 20 independent 512x512 alpha=2 freely decaying runs
(alpha-beta_turbulence/JWS_decaying_ensemble.py, seeds 1-20, same setup as output_0090.nc).

Usage: python3 extract_alpha_ensemble.py
"""
import numpy as np
import netCDF4 as nc

run_dir = "/expanse/lustre/projects/cit197/jskinner1/AB_turb/data/decay_ensemble"
alpha = "2.0"
output_idx = 90   # t ~ 917, same time as the original output_0090.nc
num_seeds = 20
out_npz = f"./JWS_turb_data/alpha-turb/alpha{alpha}_vorticity_{num_seeds}_independent_runs_output{output_idx:04d}.npz"

vort, times = [], []
for s in range(1, num_seeds + 1):
    fn = f"{run_dir}/A{alpha}-B0.0-N512-S{s}/output_{output_idx:04d}.nc"
    with nc.Dataset(fn) as f:
        vort.append(np.asarray(f["z"][:]))
        times.append(float(f["time"][0]))
    print(f"seed {s:2d}: t = {times[-1]:.1f}, std = {vort[-1].std():.4f}")
vort = np.stack(vort)
times = np.array(times)

corr = np.corrcoef(vort.reshape(num_seeds, -1))
off_diag = np.abs(corr[~np.eye(num_seeds, dtype=bool)])
print(f"Off-diagonal |r|: mean {off_diag.mean():.3f}, max {off_diag.max():.3f}")

np.savez_compressed(out_npz, vorticity=vort, time=times, seeds=np.arange(1, num_seeds + 1),
                    source=run_dir, alpha=float(alpha), output_idx=output_idx)
print(f"Saved {out_npz}  shape {vort.shape}")
