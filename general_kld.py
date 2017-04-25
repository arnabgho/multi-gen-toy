import numpy as np
import os
import matplotlib.pyplot as plt
import sys, getopt
from scipy.stats.kde import gaussian_kde
from scipy.stats import entropy
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

        inp_all=inp_all[ np.random.choice(inp_all.shape[0],out_all.shape[0],replace=False)]
        inp_out=np.concatenate((inp_all,out_all),axis=0)
        hist_temp,dummy=np.histogram(inp_out,bins='auto')
        num_bins=hist_temp.shape[0]
        inp_hist_t,dummy=np.histogram(inp_all,bins=num_bins,range=(inp_all.min(),inp_all.max()))
        out_hist_t,dummy=np.histogram(out_all,bins=num_bins,range=(inp_all.min(),inp_all.max()))
        inp_hist=[]
        out_hist=[]
        for i in xrange(inp_hist_t.shape[0]):
            if inp_hist_t[i]<1:
                continue
            inp_hist.append(inp_hist_t[i])
            out_hist.append(out_hist_t[i])
        with open(folder+'/'+str(epoch)+'/'+'stats_kld.txt','w') as f:
            f.write(str(entropy(out_hist,qk=inp_hist)))

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
