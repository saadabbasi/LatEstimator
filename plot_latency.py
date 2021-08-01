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
exp_val = latencies[:,1]#*1e3 # measured
theo_val = latencies[:,0]#*1e3 # summed

fig, ax = plt.subplots(1,1)
ax.scatter(theo_val, exp_val,s=5,color='orange')
ax.set_xlabel('Summed Latency (s)')
ax.set_ylabel('Direct Measurement (s)')
xmin = latencies[:,0].min()#*1e3
xmax = latencies[:,0].max()#*1e3
xlin = np.linspace(0,50,100)
ax.plot(xlin,xlin,0.5)
ax.plot(xlin,margin_line(0.1,exp_val,xlin),linestyle='--',color='gray')
ax.plot(xlin,margin_line(-0.1,exp_val,xlin),linestyle='--',color='gray')
ax.plot(xlin,margin_line(0.2,exp_val,xlin),linestyle='--',color='gray')
ax.plot(xlin,margin_line(-0.2,exp_val,xlin),linestyle='--',color='gray')
ax.set_xlim(exp_val.min(), exp_val.max())
ax.set_ylim(exp_val.min(), exp_val.max())
ax.set_aspect('equal')

plt.savefig(args.output_fname)