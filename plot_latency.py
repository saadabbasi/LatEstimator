import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import argparse

def percent_error(exp_value, theo_value):
    error = (exp_value - theo_value) / theo_value
    # theo_value * error + theo_value = exp_value

def margin_line(error, exp_val, theo_val):
    exp_val = theo_val * error + theo_val
    return exp_val

parser = argparse.ArgumentParser()
parser.add_argument("-o",dest="output_fname")
parser.add_argument("-i",dest="input_fname")
args = parser.parse_args()

latencies = genfromtxt(args.input_fname, delimiter=',',skip_header=True)
exp_val = latencies[:,2]*1e3 # measured
theo_val = latencies[:,1]*1e3 # summed

fig, ax = plt.subplots(1,1)
ax.scatter(theo_val, exp_val,s=5,color='orange')
ax.set_xlabel('Summed Latency (ms)')
ax.set_ylabel('Direct Measurement (ms)')
xmin = min(exp_val.min(), theo_val.min())-0.05
xmax = max(exp_val.max(), theo_val.max())+0.05
xlin = np.linspace(0,100,100)
ax.plot(xlin,xlin,0.5)
ax.plot(xlin,margin_line(0.1,exp_val,xlin),linestyle='--',color='gray')
ax.plot(xlin,margin_line(-0.1,exp_val,xlin),linestyle='--',color='gray')
ax.plot(xlin,margin_line(0.2,exp_val,xlin),linestyle='--',color='gray')
ax.plot(xlin,margin_line(-0.2,exp_val,xlin),linestyle='--',color='gray')
ax.set_xlim(xmin,xmax)
ax.set_ylim(xmin,xmax)
ax.set_aspect('equal')

plt.savefig(args.output_fname)