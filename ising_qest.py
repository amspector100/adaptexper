import os
import sys
import time
import datetime
try:
	import knockadapt
except ImportError:
	# This is super hacky, will no longer be necessary after development ends
	file_directory = os.path.dirname(os.path.abspath(__file__))
	parent_directory = os.path.split(file_directory)[0]
	knockoff_directory = parent_directory + '/knockadapt'
	sys.stdout.write(f'Knockoff dir is {knockoff_directory}\n')
	sys.path.insert(0, os.path.abspath(knockoff_directory))
	import knockadapt

# We have to call the glasso package from R
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

def main():

	# Simulate a ton of Ising data
	seed = 110
	n = 10000
	p = 625
	np.random.seed(seed)
	time0 = time.time()
	X,_,_,Q,V = knockadapt.graphs.sample_data(
	    n=n, p=p, x_dist='gibbs', method='ising',
	)
	np.fill_diagonal(Q, 1)
	print(f"Took {time.time() - time0} to sim data")

	# Construct sparsity pattern
	sparsity = []
	for i in range(p):
	    for j in range(i):
	        if Q[i,j] == 0:
	        	# Remember R is 1 indexed 
	        	# \_O_/ this took me a while to figure out
	            sparsity.append((i+1,j+1))
	sparsity = np.array(sparsity)

	# Push to R
	Vr = numpy2ri.py2rpy(V)
	sparsityr = numpy2ri.py2rpy(sparsity)

	# Estimate precision matrix using graphical lasso
	glasso = importr('glasso')
	Vglasso = glasso.glasso(Vr, rho=0.01, nobs=n, zero=sparsityr)
 
	# Extract output and enforce sparsity
	Qest = np.asarray(Vglasso[1])
	Qest[Q == 0] = 0
	Vest = knockadapt.utilities.chol2inv(Qest)
	del V
	del Q

	# Save to output
	vfname = 'vout.txt'
	np.savetxt(vfname, Vest)
	qfname = 'qout.txt'
	np.savetxt(qfname, Qest)


if __name__ == '__main__':
	main()