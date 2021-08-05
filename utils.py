import time
import numpy as np

def latency(net, input_v, N=100, use_gpu=False):
    lat = []
    if use_gpu: 
        net.to('cuda:0')
        input_v = input_v.to('cuda:0')
    for n in range(N):
        start = time.time()
        y = net(input_v)
        end = time.time()
        lat.append(end-start)
    
    lat = np.array(lat)
    return lat
    # print(f"Min. Latency (ms): {lat.min()*1e3:.2f}, Std dev: {lat.std()*1e3:.2f}")