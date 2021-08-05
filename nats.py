from nats_bench import create
import xautodl.models
import torch
import time
import numpy as np
import torch.nn as nn
from torchsummary import summary
import xautodl.models.cell_infers
from tqdm import tqdm
np.random.seed(42)

def latency(net, input_v, N=100, use_gpu=False):
    lat = []
    if use_gpu: 
        net.to('cuda:3')
        input_v = input_v.to('cuda:3')
    for n in range(N):
        start = time.time()
        y = net(input_v)
        end = time.time()
        lat.append(end-start)
    
    lat = np.array(lat)
    return lat

def random_arch_idx(n_idx):
    return np.random.choice(np.arange(15625),n_idx)

class LUT:
    def __init__(self, C=16, N=5, w=32, h=32, iters = 100, use_gpu = False):
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        self.use_gpu = use_gpu
        self.iters = iters
        self.n_classes = 10
        self.h = h
        self.w = w
        self.N = N
        self.inC = C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        self.resblock1 = xautodl.models.cell_operations.ResNetBasicblock(self.inC, self.inC*2, 2, True) # 16->32
        self.resblock2 = xautodl.models.cell_operations.ResNetBasicblock(self.inC*2, self.inC*4, 2, True) #32->64
        self.lastact = nn.Sequential(nn.BatchNorm2d(self.inC*4), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.inC*4, self.n_classes)

        self._measure_static_block_latency(224,224)

    def _measure_static_block_latency(self,h,w):
        self.LUT = {'input': 0.0, 'resblock1': 0.0, 'resblock2': 0.0, 'cell1': 0.0,
                    'cell2': 0.0, 'cell3': 0.0, 'pool':0.0}

        self.LUT['input'] = latency(self.stem,torch.randn(1,3,w,h),N=self.iters, use_gpu=self.use_gpu).mean()
        self.LUT['resblock1'] = latency(self.resblock1,torch.randn(1,self.inC,w,h),N=self.iters, use_gpu=self.use_gpu).mean()
        self.LUT['resblock2'] = latency(self.resblock2,torch.randn(1,self.inC*2,w//2,h//2),N=self.iters, use_gpu=self.use_gpu).mean()
        self.LUT['pool'] = latency(self.global_pooling,torch.randn(1,self.inC*4,w//4,h//4),N=self.iters, use_gpu=self.use_gpu).mean()
        self.LUT['lastact'] = latency(self.lastact,torch.randn(1,self.inC*4,w//4,h//4),N=self.iters, use_gpu=self.use_gpu).mean()
        self.LUT['classifier'] = latency(self.classifier,torch.randn(1,self.inC*4),N=self.iters, use_gpu=self.use_gpu).mean()

    def update_cell_latency(self, genotype):
        cell = xautodl.models.cell_infers.cells.InferCell(genotype, self.inC, self.inC, 1)
        self.LUT['cell1'] = latency(cell, torch.randn(1,self.inC,self.w,self.h),N=self.iters, use_gpu=self.use_gpu).mean()*self.N
        cell = xautodl.models.cell_infers.cells.InferCell(genotype, self.inC*2, self.inC*2, 1)
        self.LUT['cell2'] = latency(cell, torch.randn(1,self.inC*2,self.w//2,self.h//2),N=self.iters, use_gpu=self.use_gpu).mean()*self.N
        cell = xautodl.models.cell_infers.cells.InferCell(genotype, self.inC*4, self.inC*4, 1)
        self.LUT['cell3'] = latency(cell, torch.randn(1,self.inC*4,self.w//4,self.h//4),N=self.iters, use_gpu=self.use_gpu).mean()*self.N

    def total_latency(self):
        return sum(self.LUT.values())

arch_idxs = random_arch_idx(5000)
use_gpu = True
the_lut = LUT(iters=25)
x = torch.randn(1,3,32,32)
api = create('/home/saad/datasets/nats/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
with open("nasbench201-proto-gpu.txt", "w") as f:
    f.write("idx,Summed Lat,Measured Lat\n")
    for idx in tqdm(arch_idxs):
        config = api.get_net_config(idx, 'cifar10')
        the_lut.update_cell_latency(xautodl.models.cell_searchs.CellStructure.str2structure(config['arch_str']))
        network = xautodl.models.get_cell_based_tiny_net(config)
        measured = latency(network,x,N=25, use_gpu=use_gpu).mean()
        computed = the_lut.total_latency()
        f.write(f"{computed},{measured}\n")
        # print(idx, latency(network,x).mean()*1e3, the_lut.total_latency()*1e3)