from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import math
#import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ngen', type=int, default=3, help='number of generators')
parser.add_argument('--ndata', type=int, default=100000, help='number of data per epoch')
parser.add_argument('--ncentres', type=int, default=6, help='number of centres')
parser.add_argument('--std_dev', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--nz', type=int, default=3, help='size of the latent z vector')
parser.add_argument('--batchSize', type=int, default=32, help='size of a batch')
parser.add_argument('--ndim', type=int, default=2, help='Dimenstion to generate')
parser.add_argument('--R', type=int, default=5, help='Radius of the circle')
parser.add_argument('--nvis', type=int, default=3, help='Number of samples to be visualized')
parser.add_argument('--save_freq', type=int, default=1, help='How frequently to save learned model')
parser.add_argument('--exp_name', default='3gen/', help='Where to export the output')
parser.add_argument('--niter', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--batchnorm', type=bool, default=True, help='Whether to do batchnorm')	#TODO if bool is correct

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

ngen=int(opt.ngen)
ndata=int(opt.ndata)
ncentres=int(opt.ncentres)
std_dev=float(opt.std_dev)
nz=int(opt.nz)
ndim=int(opt.ndim)
R=int(opt.R)
nvis=int(opt.nvis)
save_freq=int(opt.save_freq)
real_label=ngen+1
fake_labels=torch.linspace(1,ngen,ngen)	#TODO: No need to have numpy here?

G=[]
G[0] = nn.Sequential(
			nn.Linear(3,128),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Linear(128,128),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Linear(128,ndim)
        )

for i in range(1,ngen):
	G[i] =  G[0].clone()

netD=nn.Sequential(
			nn.Linear(ndim,128),
			nn.ReLU(),
			nn.Linear(128,ngen+1)
        )
#TODO: Batch norm should be here as well
criterion = nn.CrossEntropyLoss()
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG =[]
for i in range(ngen):	#TODO:Can be problematic, any way of using model_utils.combine_all_parameters?
	optimizerG[i] = optim.Adam(G[i].parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

noise = torch.FloatTensor(opt.batchSize,nz)
input = torch.FloatTensor(opt.batchSize,ndim)
label = torch.FloatTensor(opt.batchSize)
fake_cache = torch.FloatTensor(ngen,opt.batchSize,ndim)
vis = torch.FloatTensor(nvis*opt.batchSize,ndim)
real = torch.FloatTensor(opt.batchSize,ndim)
randints = torch.FloatTensor(opt.batchSize)
#parametersD, gradParametersD = netD:getParameters()
#parametersG, gradParametersG = model_utils.combine_all_parameters(G)

for epoch in range(opt.niter):
	for iter in range(ndata/opt.batchSize):
		errD_total = 0
		for i in range(ngen):
			netD.zero_grad() #TODO: should I get it out of for loop
			randints.random_(1,ncentres)
			for j in range(1,opt.batchSize+1):
				k=randints[j]
				real[j][1]=torch.normal(0,std_dev)+R*math.cos((2*k*math.pi)/ncentres)
				real[j][2]=torch.normal(0,std_dev)+R*math.sin((2*k*math.pi)/ncentres)
			end
			input.copy_(real)
			label.fill_(real_label)
			output=netD.forward(input)
			errD_real=criterion(output,label)
			errD_real.backward()

			noise.normal_(0,1)
			fake=G[i].forward(noise)
			fake_cache[i+1].copy_(fake)
			input.copy_(fake)
			label.fill_(fake_labels[i+1])
			output=netD.forward(input)
			errD_fake=criterion(output,label)
			errD_fake.backward()
			errD = errD_real+errD_fake
			errD_total = errD_total + errD
			optimizerD.step()	#TODO: should I get it out of for loop, but then will it use errD, how will effect of gradients change (as gradients change in the for loop)

		label.fill_(real_label)
		errG_total=0
		for i in range(ngen):
			G[i].zero_grad()
			#output=netD.forward(G[i].output) #TODO: try it
			output=netD.forward(fake_cache[i+1])
			errG=criterion(output,label)
			errG.backward()
			optimizerG.step()
			errG_total=errG_total+errG
#TODO variable