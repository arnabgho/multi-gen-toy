import numpy as np
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--mode',type=str,default='out')
args=parser.parse_args()
colors = ['b', 'c', 'y', 'm', 'r']

if args.mode=='out':
    inp = np.loadtxt('out.txt', delimiter=' ')
else:
    inp = np.loadtxt('input.txt', delimiter=' ')
#plt.scatter(inp[:, 1], inp[:, 2])
if max(inp[:,0])>0:
    bl = int(inp.shape[0]/max(inp[:,0]))
else:
    bl=inp.shape[0]
lo = plt.scatter(inp[0:bl, 1], inp[0:bl, 2], color=colors[0], alpha=0.5)
ll = plt.scatter(inp[bl:(bl*2), 1], inp[bl:(bl*2), 2], color=colors[1], alpha=0.5)
l = plt.scatter(inp[(bl*2):(bl*3), 1], inp[(bl*2):(bl*3), 2], color=colors[2], alpha=0.5)
#plt.legend((lo, ll, l),
#           ('G1', 'G2', 'G3'),
#           scatterpoints=1,
#           loc='upper right')

#plt.show()

plt.savefig('output.png')

plt.close()
