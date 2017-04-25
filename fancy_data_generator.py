from optparse import OptionParser

parser=OptionParser()
parser.add_option('--folder', default='datasets',help='The dataset directory')
parser.add_option('--data_name',default='test',help='The name of the data folder')
parser.add_option('--out_name',default='input.txt',help='The name of the samples folder')
parser.add_option('--h5_filename',default='data.h5',help='The h5 file containing the samples')
parser.add_option('--num_samples',default=768000,help='The number of samples used')
parser.add_option('--dataset',default='moons',help='The dataset to be used')

(options,args)=parser.parse_args()
import os
import h5py
import numpy as np
from sklearn import datasets
import torch

if options.dataset=='circles':
    data,junk=datasets.make_circles(n_smaples=options.num_samples)
elif options.dataset=='moons':
    data,junk=datasets.make_moons(n_samples=options.num_samples)

data=np.append(np.zeros((data.shape[0],1)),data,axis=1)

f=h5py.File(os.path.join(os.getcwd(),options.folder,options.data_name,options.h5_filename),'w')
f.create_dataset('data',data=data)
f.close()
np.savetxt( os.path.join(os.getcwd(),options.folder,options.data_name,options.out_name),data,delimiter='\t' )


