import numpy as np
import os
import matplotlib.pyplot as plt
import sys, getopt


binSize = 10
colors = ['b', 'c', 'y', 'm', 'r']
files = ['input', 'out']
outFiles = ['CrealInput', 'CgeneratedOutput']

def Main(argv):
	try:
		folder = argv[0]
		epoch = argv[1]
	except:
		print('python histPlotter.py folderName epochName')

	for Id in len(files):
		inp = np.loadtxt(folder + '/' + str(epoch) + '/' + files[Id] + '.txt', delimiter=' ')
		numHist = max(inp[:,0])
		if (int(numHist) == 0):
			numHist = 1
		lenHist = int(inp.shape[0]/numHist)
		highest = max(inp[:,1])
		lowest = min(inp[:,1])
		bins = np.zeros(binSize + 1)
		binWidth = (highest - lowest) / binSize
		for j in range(binSize + 1):
			bins[j] = lowest + binWidth * j

		i = 0
		totalHist = []
		height = np.zeros(binSize)
		for j in range(int(numHist)):
			hist = np.histogram(inp[i:(i+lenHist),1], bins=binSize)
			totalHist = totalHist + [plt.bar(bins[:-1], hist[0], binWidth, bottom=height, color=colors[j])]
			height = height + hist[0]
			i = i + lenHist

		plt.plot(bins[:-1], height, '-o')
		plotted = []
		generators = []

		if files[Id] == 'out':
			for i in range(len(totalHist)):
				plotted = plotted + [totalHist[i][0]]
				generators = generators + ['G_' + str(i + 1)]

			plt.legend(tuple(plotted), tuple(generators))

		plt.savefig(folder + '/' + str(epoch) + '/' + outFiles[Id] + '.png')
		plt.close()

def Runner(argv):
	try:
		folder = argv[0]
	except:
		print('python histPlotter.py folderName')

	for i in range(1000):
		Main([folder, str(i + 1)])

if __name__=="__main__":
	Main(sys.argv[1:])
	# Runner(sys.argv[1:])
