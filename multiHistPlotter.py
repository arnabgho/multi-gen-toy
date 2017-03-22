import numpy as np
import os
import matplotlib.pyplot as plt
import sys, getopt

numBins=1000
binSize = 10
colors = ['b', 'c', 'y', 'm', 'r']
files = ['input', 'out']
outFiles = ['realInput', 'generatedOutput']

def Main(argv):
	try:
		folder = argv[0]
		epoch = argv[1]
	except:
		print('python multiHistPlotter.py folderName epochName')

        inp = np.loadtxt(folder+'/'+str(epoch)+'/'+'input.txt',delimiter=' ')
        out = np.loadtxt(folder+'/'+str(epoch)+'/'+'out.txt',delimiter=' ')

        inp_all=inp[:,1]
        ngen=max(out[:,0]).astype(int)
        num_out=out.shape[0]/ngen
        for i in xrange(1,ngen+1):
        	out_all=out[ num_out*(i-1):num_out*(i)-1,1 ]
        	minim=min(np.concatenate((out_all,inp_all),axis=0))
        	maxim=max(np.concatenate((out_all,inp_all),axis=0))
        	bins=np.linspace(minim,maxim,numBins)
		plt.hist(inp_all, bins, alpha=1, label='input',color='b'  )
		plt.hist(out_all, bins, alpha=0.1, label='G'+str(i),color='b' )
<<<<<<< HEAD
		hist = np.histogram(out_all, bins = bins)
		#plt.plot((bins[:-1] + bins[1:])/2, hist[0], '-o')
=======
>>>>>>> 47bdb2e615fa058e1c0df16790c1a89214563a32
		plt.legend(loc='upper right')
		#plt.show()
		plt.savefig(folder + '/' + str(epoch) + '/' + 'G_'+str(i) + '.png')
		plt.close()
        out_all=out[ :,1 ]
        minim=min(np.concatenate((out_all,inp_all),axis=0))
        maxim=max(np.concatenate((out_all,inp_all),axis=0))
        bins=np.linspace(minim,maxim,numBins)
	plt.hist(inp_all, bins, alpha=1, label='input',color='b'  )
	plt.hist(out_all, bins, alpha=0.1, label='G'+'_all',color='b' )
	plt.legend(loc='upper right')
	#plt.show()
	plt.savefig(folder + '/' + str(epoch) + '/' + 'G_all' + '.png')
	plt.close()

#
#		plt.savefig(folder + '/' + str(epoch) + '/' + outFiles[Id] + '.png')

#	for Id in len(files):
#		inp = np.loadtxt(folder + '/' + str(epoch) + '/' + files[Id] + '.txt', delimiter=' ')
#		numHist = max(inp[:,0])
#		if (int(numHist) == 0):
#			numHist = 1
#		lenHist = int(inp.shape[0]/numHist)
#		highest = max(inp[:,1])
#		lowest = min(inp[:,1])
#		bins = np.zeros(binSize + 1)
#		binWidth = (highest - lowest) / binSize
#		for j in range(binSize + 1):
#			bins[j] = lowest + binWidth * j
#
#		i = 0
#		totalHist = []
#		height = np.zeros(binSize)
#		for j in range(int(numHist)):
#			hist = np.histogram(inp[i:(i+lenHist),1], bins=binSize)
#			totalHist = totalHist + [plt.bar(bins[:-1], hist[0], binWidth, bottom=height, color=colors[j])]
#			height = height + hist[0]
#			i = i + lenHist
#
#		plotted = []
#		generators = []
#
#		if files[Id] == 'out':
#			for i in range(len(totalHist)):
#				plotted = plotted + [totalHist[i][0]]
#				generators = generators + ['G_' + str(i + 1)]
#
#			plt.legend(tuple(plotted), tuple(generators))
#
#		plt.savefig(folder + '/' + str(epoch) + '/' + outFiles[Id] + '.png')
#		plt.close()

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
