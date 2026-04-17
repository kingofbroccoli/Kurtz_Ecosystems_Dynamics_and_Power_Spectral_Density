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

def marchenko_pastur(x, lplus, lminus, gamma, T):
    return np.sqrt((lplus-x)*(x-lminus)) / (2*gamma*T*np.pi*x)

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
T = 1e-3
N_step = 10000 # 10000 int(tmax / dt)

gamma = N / N_step # (N_step - 6000)
mu = -0.5
xbar = 1.0 / (1.0 - mu)
lplus = (1.0 + np.sqrt(gamma))**2 * T  
lminus = (1.0 - np.sqrt(gamma))**2 * T
outlier = - T * (mu - gamma*mu + gamma*T) / (mu - 1) / mu

fig, ax = plt.subplots()
x = np.linspace(lminus, lplus, 1000)
ax.plot(x, np.sqrt((lplus-x)*(x-lminus)) / (2*gamma*T*np.pi*x), 'red', lw=1.20)
ax.set_ylim(bottom=0)
draw_dashed_line(ax, T, 'v', 'blue')
draw_dashed_line(ax, lplus, 'v', 'red')
draw_dashed_line(ax, lminus, 'v', 'red')
draw_dashed_line(ax, T*xbar, 'v', 'green')
draw_dashed_line(ax, outlier, 'v', 'orange')
print(outlier)
plot_path = f"Marchenko_Pastur_N{N}_Nstep_{N_step}_mu{mu}_T{T}.pdf"
fig.savefig(plot_path)