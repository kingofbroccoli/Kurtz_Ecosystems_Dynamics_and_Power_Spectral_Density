import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    #"figure.constrained_layout.use" : True,
    "figure.autolayout" : True,
    "figure.titlesize" : 20,
    "axes.labelsize" : 20,
    "legend.fontsize" : 15,
    "xtick.labelsize" : 15,
    "xtick.major.size" : 5,
    "ytick.labelsize" : 15,
    "ytick.major.size" : 5,
    })

N = 1000
c = '50.00'
sigma = "0.071" # "1e-05" "0.071"
volume_omega = "1.0" # "4.0" "9.0" "16.0" "100.0" "400.0" "40000.0"
dt = 0.01
save_period = 20
N_step = 5000 # 10000 int(tmax / dt)
ia_label = "HalfNormal"
single_species_to_draw = 5
binary_file = True
cmap = plt.get_cmap('Blues')

# 100,000 points per segment means we average ~18 overlapping chunks per time series.
# This trades some frequency resolution for a much smoother, cleaner curve.
#num_pts_per_seg = 50000
num_pts_per_seg = 2500
#overlap = num_pts_per_seg // 2  # 50% overlap è standard

if binary_file:
    filename = f"Linearised_Trajectories_N{N}_c{c}_{ia_label}_{sigma}_omega{volume_omega}_dt{dt}.bin"
    # Map the binary file using SSD
    data = np.memmap(filename, dtype=np.float64, mode='r', shape=(N_step, N))
    print(data.shape)
else:
    filename = f"Trajectories_N{N}_{ia_label}.txt"
    data = np.loadtxt(filename, dtype=np.float64)
    print(data.shape)

psd_sum = None
frequencies = None

print("Starting Welch PSD processing over SSD...")

for i in range(N):
    # Extract the full time series for element i
    time_series = data[:, i]
    mean_abund = time_series.mean()
    print(mean_abund)

    #f, Pxx = welch(time_series, fs=1.0/dt, nperseg=num_pts_per_seg, detrend='constant')

    # Manually subtract the global mean to remove the trivial DC peak
    ts_centered = time_series - time_series.mean()
    # Compute the PSD using Welch's method.
    # fs=1.0 assumes 1 sample = 1 unit of time. You can change this to 1/dt.
    f, Pxx = welch(ts_centered, fs=1.0/(dt*save_period), nperseg=num_pts_per_seg, detrend=False) # noverlap=overlap
 
    # On the first iteration, set up our tracking arrays
    if psd_sum is None:
        psd_sum = np.zeros_like(Pxx)
        omega = 2*np.pi*f  # Keep the frequency axis for plotting later
        traj_av = np.zeros_like(time_series)
        single_x = np.zeros((single_species_to_draw, N_step))
        single_psd = np.zeros((single_species_to_draw, Pxx.size))
    
    # Add to our running sum
    psd_sum += Pxx
    traj_av += time_series 
    if i < single_species_to_draw:
        single_x[i] = time_series
        single_psd[i] = Pxx

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {N} time series...")

#n_segments = (len(ts_centered) - num_pts_per_seg) // (num_pts_per_seg - overlap) + 1
#print(f"Number of segments per time series: {n_segments}")

# Compute the final average PSD across all species
avg_psd = psd_sum / N
traj_av /= N
# Convert SciPy's one-sided PSD (per Hz) to Physics two-sided PSD (per rad/s) - NON CREDO SERVA
avg_psd_physics = avg_psd / (4 * np.pi)

print(f"Omega spacing: {omega[0]} {omega[1]} {omega[2]}")
print("Done! You now have a clean, averaged PSD.")

fig = plt.figure(figsize=(6.4*2, 4.8))
gs = fig.add_gridspec(1, 2)#, hspace=3.3, wspace=0.09, top = 0.988, bottom = 0.09, left = 0.072, right = 0.985)
ax = gs.subplots() # (sharex='col', sharey='row')
time = np.arange(0, N_step)*(dt*save_period)
ax[0].plot(time, traj_av, color='orange')
ax[1].plot(omega, avg_psd, color='orange', label='average')
for i in range(single_species_to_draw):
    colour = cmap(0.2 + (i / (single_species_to_draw - 1)) * 0.8)
    ax[0].plot(time, single_x[i], color=colour)
    ax[1].plot(omega, single_psd[i], color=colour, label=f'species ${i+1}$')
# Time series plot
#ax[0].set_yscale('log')
ax[0].set_xlim(left=0, right=1000)
ax[0].set_xlabel('$t$')
ax[0].set_ylabel(f'$\\langle x(t) \\rangle$')
ax[0].grid(True, alpha=0.3, which='both')
# PSD plot
#ax[1].set_xscale('log')
#ax[1].set_yscale('log')
ax[1].set_xlim(left=0.0, right=1.50)
ax[1].set_ylim(bottom=0.0)
ax[1].grid(True, alpha=0.3, which='both')
ax[1].set_xlabel('$\\omega$')
ax[1].set_ylabel('$\\Phi(\\omega)$')
ax[1].legend(loc='upper right', shadow=True)
plot_path = f"Welch_Method_Linearised_GLV_PSD_Trajectories_N{N}_c{c}_{ia_label}_{sigma}_omega{volume_omega}.pdf"
fig.savefig(plot_path)
plt.close(fig)
