import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path

def draw_dashed_line(ax, value, align, color="#000000", ls=(0, (5, 10)), lw=1.0):
    if align == 'v':
        ylim = ax.get_ylim()
        y_line = np.linspace(ylim[0], ylim[1], 2)
        value_line = np.full(2, value)
        ax.plot(value_line, y_line, label=None, linestyle=ls, linewidth=lw, color=color, zorder=0)
        ax.set_ylim(ylim)
    elif align == 'h':
        xlim = ax.get_xlim()
        x_line = np.linspace(xlim[0], xlim[1], 2)
        value_line = np.full(2, value)
        ax.plot(x_line, value_line, label=None, linestyle=ls, linewidth=lw, color=color, zorder=0)
        ax.set_xlim(xlim)
    else:
        raise ValueError

def eigenvalues_numpy_from_matrix(matrix):
    w = np.linalg.eigvals(matrix)
    sorted_indexes = np.argsort(w)[::-1] 
    w = w[sorted_indexes]
    return w

def real_eigenvalues_histogram(ax, w, histo_dir, histo_data_file, delta, label=None, color='#9999ff'):
    # Parameters
    my_max = w.max()
    my_min = w.min()
    lwd = 0#0.001  
    bins_edge = [my_min]
    while(bins_edge[-1] < my_max):
        bins_edge.append(bins_edge[-1] + delta)
    N_bins = len(bins_edge) - 1
    hist, tmp_bins_edge = np.histogram(w, bins_edge) # Devo usare tmp_bins_edge perché l'oggetto ritornato è un array mentre mi sta comodo che bins_edge sia una lista
    # Histogram details
    N_tot = hist.sum()
    inv = 1.0/(delta*N_tot)
    density_hist = np.array([h*inv for h in hist])
    wd = np.full(density_hist.size, delta)
    # Saving histo data
    histo_file = histo_dir / histo_data_file
    with open(histo_file, 'w') as hf:
        for l in (bins_edge[:-1], density_hist, wd): # Mi sta più comodo non salvare l'ultimo elemento per la lettura con numpy. Eventualmente in seguito trova un altro modo per leggere
            for element in l:
                hf.write(f"{element}\t")
            hf.write("\n")
    # Plotting Histogram
    ax.bar(bins_edge[:-1], density_hist, wd, align='edge', color=color, edgecolor='#000000', linewidth=lwd, alpha=0.75, label=label)
    return

plt.rcParams.update({
    "text.usetex": True,
    "font.family" : "serif",
    "font.serif" : ["Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
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

N = 100
c = '100.00' # '0.00' '200.00'
mu = '0.5'
sigma = '0.0' # "1e-05" "0.071"
pA = '0.0'
pM = '0.0'
T_label = "1e-3" # "4.0" "9.0" "16.0" "100.0" "400.0" "40000.0"
dt = 0.01
save_period = 500
#dt = 5e-2
#save_period = 100
dtsam = int(save_period * dt)
print(dtsam)
N_step = 10000 # 10000 int(tmax / dt)
ia_label = "MixtureTruncGauss"
single_species_to_draw = 5
surv_thresh = 1e-3
binary_file = True
cmap = plt.get_cmap('Blues')

# --- Configuration ---
CHUNK_SIZE = 5000 # Number of time steps to process in RAM at once
if binary_file:
    filename = f"Trajectories_N{N}_c{c}_{ia_label}_mu{mu}_sigma{sigma}_pA_{pA}_pM_{pM}_T{T_label}_dt{dt}_dtsam_{dtsam}.bin"
    # Map the binary file using SSD
    data = np.memmap(filename, dtype=np.float64, mode='r', shape=(N_step, N))
    print(data.shape)
else:
    filename = f"Trajectories_N{N}_c{c}_{ia_label}_mu{mu}_sigma{sigma}_pA_{pA}_pM_{pM}_T{T_label}_dt{dt}_dtsam_{dtsam}.txt"
    data = np.loadtxt(filename, dtype=np.float64)
    print(data.shape)

# Map the binary file (assuming double precision floats from C)
data = np.memmap(filename, dtype=np.float64, mode='r', shape=(N_step, N))

# --- Initialize tracking arrays ---
# For the uncentered correlation matrix <x_i x_j>
corr_matrix_sum = np.zeros((N, N), dtype=np.float64)

# For the mean <x_i> (needed if you want the centered covariance matrix)
sum_vec = np.zeros(N, dtype=np.float64)

print(f"Computing N x N spatial correlation matrix in chunks of {CHUNK_SIZE} steps...")

# --- Process the data in chunks ---
for start_idx in range(0, N_step, CHUNK_SIZE):
    # Get the end index for this chunk (handles the final chunk if N_step isn't perfectly divisible)
    end_idx = min(start_idx + CHUNK_SIZE, N_step)
    
    # Load the chunk into RAM. Shape: (CHUNK_SIZE, N)
    chunk = data[start_idx:end_idx, :]
    
    # 1. Update the mean tracking
    sum_vec += np.sum(chunk, axis=0)
    
    # 2. Update the correlation matrix tracking
    # chunk.T @ chunk is a highly optimized way to compute the sum of x_i * x_j for all pairs
    corr_matrix_sum += chunk.T @ chunk
    
    # Print progress
    if start_idx > 0 and (start_idx % 1000) == 0:
        print(f"Processed {start_idx} / {N_step} time steps...")

# --- Final Computations ---
# 1. The Empirical Spatial Correlation Matrix: <x_i x_j>
empirical_corr_matrix = corr_matrix_sum / N_step

# 2. The Mean Vector: <x_i>
mean_vec = sum_vec / N_step
#print(mean_vec)

# 3. The Centered Covariance Matrix: <x_i x_j> - <x_i><x_j>
# np.outer efficiently computes the matrix of <x_i>*<x_j>
cov_matrix = empirical_corr_matrix - np.outer(mean_vec, mean_vec)

print("Computation complete!")

# --- Visualization ---
# Plotting the Covariance Matrix as a heatmap

fig, ax = plt.subplots(figsize=(8, 6))

# We plot the centered covariance matrix, but you can swap this to 'empirical_corr_matrix'
cax = ax.imshow(cov_matrix, cmap='viridis', aspect='auto', origin='upper')
fig.colorbar(cax, label=f'Covariance $C_{{ij}}$')

ax.set_title("Spatial Covariance Matrix")
ax.set_xlabel(f'Element $j$')
ax.set_ylabel(f'Element $i$')

plot_path = f"Spatial_Covariance_Matrix_N{N}_c{c}_{ia_label}_mu{mu}_sigma{sigma}_pA_{pA}_pM_{pM}_T{T_label}_dt{dt}.pdf"
fig.savefig(plot_path)

# Eigenvalues

gamma = N / N_step # (N_step - 6000)
if pM == '0.0':
    fmu = -float(mu)
elif pM == '1.0':
    fmu = +float(mu)
else:
    raise ValueError(f"What pM is {pM}?")
xbar = 1.0 / (1.0 - fmu)
T = float(T_label)
lplus = (1.0 + np.sqrt(gamma))**2 * T  
lminus = (1.0 - np.sqrt(gamma))**2 * T
outlier = - T * (fmu - gamma*fmu + gamma*T) / (fmu - 1) / fmu

def mp(x, lplus, lminus, gamma, T):
    return np.sqrt((lplus-x)*(x-lminus)) / (2*gamma*T*np.pi*x)

delta = 5e-5 # 1e-2 * T
delta = 2e-2 * T
w = eigenvalues_numpy_from_matrix(cov_matrix)
print(T)
print(w.mean())
print(w.std())
histo_dir = Path.cwd()
histo_data_file = histo_dir / "Histo_Data.dat"
fig, ax = plt.subplots()
real_eigenvalues_histogram(ax, w, histo_dir, histo_data_file, delta, label=None, color='#9999ff')
# Plot MP
x = np.linspace(lminus, lplus, 1000)
ax.plot(x, np.sqrt((lplus-x)*(x-lminus)) / (2*gamma*T*np.pi*x), 'red', lw=1.20)
draw_dashed_line(ax, w.mean(), 'v', 'red')
draw_dashed_line(ax, lplus, 'v', 'red')
draw_dashed_line(ax, lminus, 'v', 'red')
draw_dashed_line(ax, T*xbar, 'v', 'green')
draw_dashed_line(ax, outlier, 'v', 'orange')
print(outlier)
plot_path = f"Spatial_Covariance_Matrix_Spectrum_N{N}_c{c}_{ia_label}_mu{mu}_sigma{sigma}_pA_{pA}_pM_{pM}_T{T_label}_dt{dt}.pdf"
fig.savefig(plot_path)