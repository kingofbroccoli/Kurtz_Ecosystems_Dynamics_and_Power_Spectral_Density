import numpy as np
from scipy.signal import welch, csd
import matplotlib.pyplot as plt

N = 2
dt = 0.01
N_step = 80000 # 100001 int(tmax / dt)
ia_label = "HalfNormal"
binary_file = False
if binary_file:
    filename = f"Trajectories_N{N}_{ia_label}.bin"
    # Map the binary file using SSD
    data = np.memmap(filename, dtype=np.float64, mode='r', shape=(N_step, N))
    print(data.shape)
else:
    filename = f"Trajectories_N{N}_{ia_label}.txt"
    data = np.loadtxt(filename, dtype=np.float64)
    print(data.shape)

num_pts_per_seg = 20000 

# Extract prey and predator time series (dropping first 10,000 steps for equilibration)
# Assuming index 0 is prey and index 1 is predator
prey = data[10000:, 0]
predator = data[10000:, 1]

# Center the data
prey_centered = prey - prey.mean()
predator_centered = predator - predator.mean()

# 1. Compute individual PSDs to see the Quasi-Cycle Peak
f, Pxx_prey = welch(prey_centered, fs=1.0/dt, nperseg=num_pts_per_seg, detrend=False)
_, Pxx_pred = welch(predator_centered, fs=1.0/dt, nperseg=num_pts_per_seg, detrend=False)

# 2. Compute the Cross-Spectral Density (CSD) between prey and predator
# CSD is a complex array. Its magnitude shows shared power, its angle shows phase lag.
f_csd, Pxy = csd(prey_centered, predator_centered, fs=1.0/dt, nperseg=num_pts_per_seg, detrend=False)

# Convert frequencies to physics notation
omega = 2 * np.pi * f
omega_csd = 2 * np.pi * f_csd

# Convert to physics scaling
S_prey = Pxx_prey / (4 * np.pi)
S_pred = Pxx_pred / (4 * np.pi)
S_xy = Pxy / (4 * np.pi)

# Extract the Phase Angle from the complex CSD
phase_angle = np.angle(S_xy)

print("Plotting Quasi-cycles and Cross-Spectrum Phase...")

fig = plt.figure(figsize=(20, 5))
gs = fig.add_gridspec(1, 4)
ax = gs.subplots()

# Plot 1: The Trajectories (Time Domain)
time_axis = np.arange(0, prey_centered.size) * dt
ax[0].plot(time_axis, prey, label='Prey', alpha=0.8)
ax[0].plot(time_axis, predator, label='Predator', alpha=0.8)
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('Abundance')
ax[0].set_xlim(0, 100) # Zoom in to see the noisy cycles
ax[0].grid(True, alpha=0.3, which='both')
ax[0].legend()

# Plot 2: The Trajectories (Time Domain)
time_axis = np.arange(0, prey_centered.size) * dt
ax[1].plot(prey[-num_pts_per_seg:], predator[-num_pts_per_seg:], label='Prey-Predator', alpha=0.8)
ax[1].set_xlabel('Prey')
ax[1].set_ylabel('Predator')
ax[1].grid(True, alpha=0.3, which='both')
ax[1].legend()

# Plot 3: Individual PSDs (The Quasi-Cycle Peak)
ax[2].plot(omega, S_prey, label='Prey PSD')
ax[2].plot(omega, S_pred, label='Predator PSD')
ax[2].set_xlabel(r'$\omega$')
ax[2].set_ylabel(r'$S(\omega)$')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].grid(True, alpha=0.3, which='both')
ax[2].legend()

# Plot 4: The Phase Angle of the CSD (The Anti-correlated Noise Signature)
ax[3].plot(omega_csd, phase_angle, color='purple')
ax[3].axhline(np.pi, color='k', linestyle='--', alpha=0.5, label=r'$+\pi$ (Anti-correlated)')
ax[3].axhline(-np.pi, color='k', linestyle='--', alpha=0.5, label=r'$-\pi$ (Anti-correlated)')
ax[3].set_xlabel(r'$\omega$')
ax[3].set_ylabel(r'Phase Difference (rad)')
ax[3].set_xscale('log')
ax[3].set_ylim(-np.pi - 0.5, np.pi + 0.5)
ax[3].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax[3].set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax[3].grid(True, alpha=0.3, which='both')
ax[3].legend()

plt.tight_layout()
plt.show()