import numpy as np
import os
import matplotlib.pyplot as plt
import sys, getopt
from scipy.stats.kde import gaussian_kde
from scipy.stats import ks_2samp
numBins=1000
binSize = 10
colors = ['b', 'c', 'y', 'm', 'r']
files = ['input', 'out']
outFiles = ['realInput', 'generatedOutput']

def Main(argv):
	try:
		folder = argv[0]
                epoch = argv[1]
                data = argv[2]
	except:
		print('python general_kdePlotter.py folderName epochName dataname')

        inp = np.loadtxt('datasets/'+str(data)+'/'+'input.txt',delimiter=' ')
        folder='datasets/'+str(data)+'/'+ folder
        out = np.loadtxt(folder+'/'+str(epoch)+'/'+'out.txt',delimiter=' ')

        inp_all=inp[:,1]
        out_all=out[:,1]
        with open(folder+'/'+str(epoch)+'/'+'stats.txt','w') as f:
            f.write(str(ks_2samp(inp_all,out_all)))

def Runner(argv):
	try:
		folder = argv[0]
	except:
		print('python multiHistPlotter.py folderName')

	for i in range(1000):
		Main([folder, str(i + 1)])

if __name__=="__main__":
	Main(sys.argv[1:])
	# Runner(sys.argv[1:])
