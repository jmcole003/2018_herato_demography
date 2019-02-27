
# Library importation
import os
import sys
import getopt
import matplotlib
matplotlib.use('Agg')

from scipy import stats
import pylab
import numpy
from numpy import array
import dadi


#Import 26 models 
import models1
import models2



#Load in data SFS
data = dadi.Spectrum.from_file('file_name.sfs'  pop_ids =['BeL','BeS'])
ns=[15,15]

#Grid points
pts_l=[35,45,55]


#Paramter bounds
upper_bound = [20, 20, 10, 10, 30, 30, 10, 2, 0.95, 0.99]
lower_bound = [0.01, 0.01, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.05, 0.8]


#Initial starting parameters
p0=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]



#Specify model to run... in this case a strict isolation (SI) model
func_ex = dadi.Numerics.make_extrap_log_func(models1.SI)

#Optimization (3 rounds)

p0 = dadi.Misc.perturb_params(p0, fold=1, lower_bound=lower_bound, upper_bound=upper_bound)

popt = dadi.Inference.optimize_anneal(p0, data, func_ex, pts_l, 
						      lower_bound=lower_bound,
						      upper_bound=upper_bound,
						      verbose=len(p0),
						      maxiter=100, Tini=50, Tfin=0, 
						      learn_rate=0.005, schedule="cauchy")

p1 = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])


popt2 = dadi.Inference.optimize_anneal(p1, data, func_ex, pts_l, 
						      lower_bound=lower_bound,
						      upper_bound=upper_bound,
						      verbose=len(p0),
						      maxiter=100/2, Tini=50/2, Tfin=0, 
						      learn_rate=0.005*2, schedule="cauchy")

p2 = (popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5], popt2[6], popt2[7], popt2[8], popt2[9])
 
popt3 = dadi.Inference.optimize_log(p2, data, func_ex, pts_l, 
						   lower_bound=lower_bound,
						   upper_bound=upper_bound,
						   verbose=len(p0),
						   maxiter=100/2)

# Computation of statistics
model = func_ex(popt3, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)
theta = dadi.Inference.optimal_sfs_scaling(model, data)
AIC = 2*len(popt3)-2*ll_opt
optim = popt3

#Print
print('Optimized log likelihood: {0}'.format(ll_opt))
print('Optimized theta: {0}'.format(theta))
print('AIC: {0}'.format(AIC))
print('Best-fit parameters: {0}'.format(optim))

#Plot Model Comparison
pylab.figure(1)
dadi.Plotting.plot_2d_comp_multinom(model, fs_obs, vmin=1, resid_range=3,
                                    pop_ids =('Pop1','Pop2'))
pylab.show()