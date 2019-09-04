# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import numpy as np
from .utils import Cat_LinReg, Cat_simulated_annealing, Cat_sample_inputs
import time

from copy import deepcopy


def Cat_BOCS_suggest(X, Y, n_Cats, order=2, SABudget=100,
					           SA_reruns=50, gibbs_time_limit=30):
	# Cat_BOCS_suggest: makes a suggestion about where to evaluate next
	#
	# ARGS
	#  X: N*categories integer matrix
	#  Y: N real values
	#  n_Cats: upper bounds of each categorical/integer argument
	#  order: the highest order polynomial term in regression model
	#  SABudget: iterations of (and SA) BOCS to perform
	#  SA_reruns: number of restarts for SA
	#  iteration_time_limit: integer, seconds per iteration
	#
	# RETURNS
	#  x_new: a integer/categorical vector

	x_vals = deepcopy(X)
	y_vals = deepcopy(Y)

	# Rescale y-values to [0, 1]
	y_vals = y_vals - y_vals.min()
	if y_vals.max()==0:
		ymax=0.01
	else:
		ymax=y_vals.max()
	y_vals = y_vals/ymax

	assert all([all(xi<n_Cats) for xi in x_vals]), "x_vals greater than number of categories"
	assert all([all(xi>=0) for xi in x_vals]), "x_vals cannot be negative"
	assert x_vals.shape[0]==len(y_vals), "x_vals and y_vals must be same length"

	# start the while loop with an invalid x_new
	x_new = x_vals[0,:]

	# to avoid getting stuck
	start_time = time.time()

	penalize = lambda x: np.any(np.all(x_vals == x, axis=1))* np.max(y_vals)

	# keep thompson sampling until x_new is not in past data
	while np.any(np.all(x_vals == x_new, axis=1)):

		# if time.time() - start_time > iteration_time_limit:
		# 	print("Hit time limit, random sampling x_new instead")
		#
		# 	# keep random sampling until x_new is not in past data
		# 	while np.any(np.all(x_vals == x_new, axis=1)):
		# 		x_new = Cat_sample_inputs(1, n_Cats).reshape((-1,))
		#
		# else:
		# train linear model
		LR = Cat_LinReg(n_Cats, order)
		LR.train(x_vals, y_vals, gibbs_time_limit)

		# define aqcuitistion function
		stat_model = lambda x: LR.surrogate_model(x, LR.alpha) + penalize(x)

		# optimize acquisition function
		SA_model = np.zeros((SA_reruns, len(n_Cats)))
		SA_obj	 = np.zeros(SA_reruns)
		for j in range(SA_reruns):
			(optModel, objVals) = Cat_simulated_annealing(stat_model, n_Cats, SABudget)
			SA_model[j,:] = optModel[-1,:]
			SA_obj[j]	  = objVals[-1]

		min_idx = np.argmin(SA_obj)
		x_new = SA_model[min_idx,:]

	return x_new.astype(int)


def Cat_BOCS1(fnoisy,f, n_Cats, n_init=None, n_evals=20, verbose=False,knownmax=0, **kwargs):
	# Miimizes f over the constrained search space of integers
	#
	# ARGS
	#  f: objective function to be minimized
	#  n_Cats: a vector of the number of categories for each input to f
	#  n_init: number of (x, f(x)) to warm start the optimization
	#  n_evals: total number of calls to f(x)
	#  verbose: boolean, execute print statements
	#  **kwargs: passed to Cat_BOCS_suggest()
	#
	# RETURNS
	#  dict: x, y, min_x, min_y
	#
##	Ammended by Paul Kent to return an array of the value of the best solution at iteration
	ARRAY2=[]
	TimerArray=[]
	TrueVals=np.zeros((0))
##
	x_vals = Cat_sample_inputs(n_init, n_Cats)
	y_vals = np.zeros((0))

	for xi in x_vals:
		y_new = fnoisy(xi)
		y_vals  = np.append(y_vals, y_new)
		TrueVals = np.append(TrueVals,-f(x_vals[np.argmin(y_vals),:]))

##
		ARRAY2 = np.append(ARRAY2,-np.min(y_vals))
##
		if(verbose):
				print("Initial sampling", len(y_vals),". x_new:", xi, ", y_new:", -y_new, ", best x:", x_vals[np.argmin(y_vals),:], ", best y:", -np.min(y_vals))

	while x_vals.shape[0] < n_evals:
##
		Start = time.time()
##
		x_new = Cat_BOCS_suggest(X=x_vals, Y=y_vals, n_Cats=n_Cats, **kwargs)

		y_new = fnoisy(x_new)

		x_vals = np.vstack([x_vals, x_new])
		y_vals = np.append(y_vals, y_new)
##
		Finish = time.time()
		ARRAY2 = np.append(ARRAY2,-np.min(y_vals))
		TrueVals = np.append(TrueVals,-f(x_vals[np.argmin(y_vals),:]))
		TimerArray = np.append(TimerArray,(Finish-Start))
##
		if verbose:
			print(x_vals.shape[0], ". x_new:", x_new, ", y_new:", -y_new,
						", best x:", x_vals[np.argmin(y_vals),:],", best y:", -np.min(y_vals))
		if knownmax!=0:
			if np.min(y_vals)==-knownmax:
				print(n_evals)
				print(x_vals.shape[0])
				for j in range(n_evals-x_vals.shape[0]):
					ARRAY2 = np.append(ARRAY2,-np.min(y_vals))
					TimerArray = np.append(TimerArray,0)
				#ARRAY2=ARRAY2.reshape(1,n_evals)
				#print(ARRAY2.shape)
				return(ARRAY2,TimerArray)
	#print (TrueVals)
	return (ARRAY2, TimerArray,TrueVals)
	#'x': x_vals,
	#'y': y_vals,
	#'min_x':x_vals[np.argmin(y_vals),:],
	#'min_y':np.min(y_vals)}



def MO_Cat_BOCS_suggest(X, Y, **kwargs):
    # Takes in past observations and returns a new x value.
    # Y may have more than one dimension, i.e. multi-objective and
    # a random projection is used.
    #
    # ARGS
    #  X: N*x_dims matrix of inputs
    #  Y: N*y_dims matrix of outputs
    #  **kwargs: passed to Cat_BOCS_suggest()
    #
    # RETURNS
    #  x_new: an x_dims array of integers to evaluate the objective next


    x_vals = np.array(deepcopy(X))
    y_vals = np.array(deepcopy(Y))

    if len(y_vals.shape)==1:
        y_vals = y_vals.reshape((-1,1))

    assert x_vals.shape[0]==y_vals.shape[0], "x and y must have equal entries"


    # normalize y values to unit square
    y_vals = y_vals - y_vals.min(axis=0)
    y_vals = y_vals/y_vals.max(axis=0)

    # pick random direction in positive quadrant
    n_objs = y_vals.shape[1]
    w = np.abs(np.random.normal(size = (n_objs, 1)))
    w = w / np.sum(w*w)

    # project y values onto random direction
    scalar_y_vals = np.matmul(y_vals, w).reshape(-1,)

    x_new = Cat_BOCS_suggest(x_vals, scalar_y_vals, **kwargs)

    return(x_new)

def MO_Cat_BOCS(f, n_Cats, n_init=None, n_evals=20, verbose=False, **kwargs):
    # Minimizes f over the constrained search space of integers
    #
    # ARGS
    #  f: objective function to be minimized
    #  n_Cats: a vector of the number of categories for each input to f
    #  n_init: number of (x, f(x)) to warm start the optimization
    #  n_evals: total number of calls to f(x)
    #  verbose: boolean, print eaqcxh iteration
    #  **kwargs: passed to Multi_Cat_BOCS_suggest
    #
    # RETURNS
    #  dict: x, y, x_pareto, y_pareto
    #

    if n_init is None:
        n_init = np.max(n_Cats)

    x_vals = Cat_sample_inputs(n_init, n_Cats)
    y_vals = f(x_vals[0,:])

    for xi in x_vals[1:]:
        y_new = f(xi)
        y_vals  = np.vstack([y_vals, y_new])
        if(verbose):
                print("Initial sampling", len(y_vals),". x_new:", xi, ", y_new:", y_new)

    while x_vals.shape[0] < n_evals:
        x_new = MO_Cat_BOCS_suggest(X=x_vals, Y=y_vals, n_Cats=n_Cats, **kwargs)
        y_new = f(x_new)

        x_vals = np.vstack([x_vals, x_new])
        y_vals = np.vstack([y_vals, y_new])

        if verbose:
            print(x_vals.shape[0], ". x_new:", x_new, ", y_new:", y_new)


    # get the non-dominated points and sort them by first column
    dominated = [np.any(np.all(y_vals<y, axis=1)) for y in y_vals]

    pareto = np.invert(dominated)
    y_pareto = y_vals[pareto, :]
    x_pareto = x_vals[pareto, :]

    A = np.argsort(y_pareto[:,0])
    y_pareto = y_pareto[A,:]
    x_pareto = x_pareto[A,:]

    return {'x': x_vals,
            'y': y_vals,
            'pareto_x':x_pareto,
            'pareto_y':y_pareto
            }


# -- END OF FILE --
