from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from hw_nas_bench_api.fbnet_models import FBNet_Infer, OPS, ConvNorm, PRIMITIVES
import argparse
import torch as torch
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm

np.random.seed(42)

def get_rand_arch():
    return np.random.choice(np.arange(9),22)

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

def generate_random_set_arch(N=2000):
    arch_configs = np.zeros((N,22),dtype=np.int8)
    for n in range(N):
        arch_configs[n,:] = get_rand_arch()

    return arch_configs

def make_LUT(N=100, use_gpu=False):
    LUT = {'input':None,
        'k3_e1':[],
        'k3_e1_g2':[],
        'k3_e3':[],
        'k3_e6':[],
        'k5_e1':[],
        'k5_e1_g2':[],
        'k5_e3':[],
        'k5_e6':[],
        'skip':[],
        'pointwise':None,
        'avgpool':None,
        'output':None}
    
    convnorm = ConvNorm(3,16,3,2)
    LUT['input'] = latency(convnorm, torch.randn(1,3,224,224),N=N,use_gpu=use_gpu)
    # imgh, imgw, cin, cout, layer_id, stride
    layer_configs = [(112,112,16,16,1,1),
                    (112,112,16,24,2,2),
                    (112,112,16,24,2,2),
                    (112,112,16,24,2,2),
                    (112,112,16,24,2,2),
                    (56,56,24,32,3,2),
                    (56,56,24,32,3,2),
                    (56,56,24,32,3,2),
                    (56,56,24,32,3,2),
                    (28,28,32,64,4,2),
                    (28,28,32,64,4,2),
                    (28,28,32,64,4,2),
                    (28,28,32,64,4,2),
                    (14,14,64,112,5,1),
                    (14,14,64,112,5,1),
                    (14,14,64,112,5,1),
                    (14,14,64,112,5,1),
                    (14,14,112,184,6,2),
                    (14,14,112,184,6,2),
                    (14,14,112,184,6,2),
                    (14,14,112,184,6,2),
                    (7,7,184,352,7,1)]
    
    if use_gpu:
        desc = "Computing LUT with GPU"
    else:
        desc = "Computing LUT with CPU"
    for op in tqdm(PRIMITIVES,desc=desc):
        for config in layer_configs:
            block = OPS[op](config[2],config[3],config[4],config[5])
            LUT[op].append(latency(block, torch.randn(1,config[2],config[0],config[1]),N=N,use_gpu=use_gpu))
    
    LUT['pointwise'] = latency(ConvNorm(352,1504,1,1),torch.randn(1,352,7,7),N=N,use_gpu=use_gpu)
    LUT['avgpool'] = latency(nn.AdaptiveAvgPool2d(1),torch.randn(1,1504,7,7),N=N,use_gpu=use_gpu)
    LUT['output'] = latency(nn.Linear(1504, 1000),torch.randn(1,1504),N=N,use_gpu=use_gpu)
    return LUT

def calculate_latency(config, LUT):
    summed_lat = LUT['input'].mean() + LUT['pointwise'].mean() + LUT['avgpool'].mean() + LUT['output'].mean()
    for layer_id, c in enumerate(config):
        op = PRIMITIVES[c]
        summed_lat += LUT[op][layer_id].mean()
    return summed_lat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="output_fname", default="temp_lat.txt")
    parser.add_argument("-it", dest="iters", type=int, default=25)
    parser.add_argument("-n", dest="N", type=int, default=2000)
    parser.add_argument("-gpu", dest="use_gpu", action="store_true")
    args = parser.parse_args()

    hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="fbnet")
    arch_configs_ref = np.load("config_ref.npz")['arch_configs_ref']
    LUT = make_LUT(N=args.iters, use_gpu=args.use_gpu)
    input_vec = torch.randn(1,3,224,224)
    if args.use_gpu:
        desc = "Measuring Latency with GPU"
    else:
        desc = "Measuring Latency with CPU"

    with open(args.output_fname,"w") as f:
        for n in tqdm(range(args.N),desc):
            arch = arch_configs_ref[n]
            summed_lat = calculate_latency(arch,LUT)
            config = hw_api.get_net_config(arch,'ImageNet')
            network = FBNet_Infer(config) # create the network from configurration
            lat = latency(network,input_vec,N=args.iters,use_gpu=args.use_gpu)
            f.write(f"{summed_lat},{lat.mean()}\n")