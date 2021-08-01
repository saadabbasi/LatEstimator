# %%
import numpy as np
from numpy import exp, genfromtxt
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

REFERNCE_CPU = "Intel Core i5"
KIWI_CPU = "Intel Core i9"
PLATYPUS_CPU = "Intel Xeon Gold"
RASPBERRY_PI = "Broadcom BCM2711"

def read_latency(fname, skip_header= True):
    latencies = genfromtxt(fname, delimiter=',',skip_header=skip_header)
    exp_val = latencies[:,1]#*1e3 # measured
    theo_val = latencies[:,0]#*1e3 # summed
    return exp_val, theo_val

def plot_correlations(ax, arr1, arr2, drawxy=False):
    r, p = pearsonr(arr1, arr2)
    pearson = AnchoredText(f"pearson corr {r:.3}",loc=2)
    xlinspace = np.linspace(0,50,100)
    ax.scatter(arr1*1e3, arr2*1e3, s=1.5, c='orange')
    ax.add_artist(pearson)
    if drawxy:
        linefrom = min([arr1.min(),arr1.min()])*1e3
        lineto = min([arr1.max(), arr2.max()])*1e3
        xyline = np.linspace(linefrom, lineto,100)
        ax.plot(xyline,xyline)
    return ax

# %%
reference_dist = "results/home-2k.txt"
exp_val_ref, theo_val_ref = read_latency(reference_dist)

kiwi_dist = "results/kiwi-2k.txt"
exp_val_kiwi, theo_val_kiwi = read_latency(kiwi_dist)


# %%
fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True)
ax1 = plot_correlations(ax1, theo_val_ref, theo_val_kiwi,True)
ax1.set_xlabel(REFERNCE_CPU+"(ms)")
ax1.set_ylabel(KIWI_CPU+"(ms)")

ax2 = plot_correlations(ax2, exp_val_ref, exp_val_kiwi,True)
ax2.set_xlabel(REFERNCE_CPU+"(ms)")
ax2.set_ylabel(KIWI_CPU+"(ms)")

# %%
plat_dist = "results/latency-plat-2k.txt"
exp_val_plat, theo_val_plat = read_latency(plat_dist)
fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True)
ax1 = plot_correlations(ax1, theo_val_ref, theo_val_plat,True)
ax1.set_xlabel(REFERNCE_CPU+"(ms)")
ax1.set_ylabel(PLATYPUS_CPU+"(ms)")

ax2 = plot_correlations(ax2, exp_val_ref, exp_val_plat,True)
ax2.set_xlabel(REFERNCE_CPU+"(ms)")
ax2.set_ylabel(PLATYPUS_CPU+"(ms)")

# %%
exp_val_rpi, theo_val_rpi = read_latency("results/latency-rpi.txt",skip_header=False)
fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True)
ax1 = plot_correlations(ax1, theo_val_ref[0:500], theo_val_rpi,True)
ax1.set_xlabel(REFERNCE_CPU+"(ms)")
ax1.set_ylabel(RASPBERRY_PI+"(ms)")

ax2 = plot_correlations(ax2, exp_val_ref[0:500], exp_val_rpi,True)
ax2.set_xlabel(REFERNCE_CPU+"(ms)")
ax2.set_ylabel(RASPBERRY_PI+"(ms)")
# %%

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# theo_val_ref = np.expand_dims(theo_val_ref,axis=-1)
# theo_val_kiwi = np.expand_dims(theo_val_kiwi,axis=-1)
# theo_val_ref = scaler.fit_transform(theo_val_ref).squeeze()
# theo_val_kiwi = scaler.transform(theo_val_kiwi).squeeze()

# fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True)
# ax1 = plot_correlations(ax1, theo_val_ref, theo_val_kiwi,True)
# %%
