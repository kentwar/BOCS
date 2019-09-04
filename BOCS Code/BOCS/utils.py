
from itertools import combinations
import numpy as np
import time

def Cat_sample_inputs(num_points, nCats):
    # Function samples the categorical input space with LHC and no repeats
    # ARGS:
    #   num_points: integer, sample size
    #   nCats: integer vector of number of categories, arbitrary length
    #
    # RETURNS
    #   samples: n_points*dim(nCats) matrix of integers

	def lhc_cats(n, cats):
		# 1 dimensional LHC
		if n>cats:
			m = np.floor(n/cats)
			A = np.arange(m*cats)%cats
			B = np.random.choice(cats, size = n%cats)
			C = np.concatenate([A, B]).astype(int)
			np.random.shuffle(C)
		else:
			C = np.random.choice(cats, size = n, replace=False)

		return C.reshape((-1, 1))


	def lhc(n):
		temp_samples = [lhc_cats(n, Ci).reshape(-1,1) for Ci in nCats]
		temp_samples = np.hstack(temp_samples)
		return temp_samples

	samples = lhc(num_points)

	# if we have need more samples than grid points, don't filter
	if num_points > np.prod(nCats):
		return samples
	else:
		samples = np.unique(samples, axis=0)

		# check to make sure there are no repeats and add new points!
		while samples.shape[0]<num_points:
			samples = np.vstack([samples, lhc(num_points-samples.shape[0])])
			samples = np.unique(samples, axis=0)

        # shuffle the samples (np.unique sorts rows)
		samples = samples.astype(int)
		A = np.argsort(np.random.uniform(size=(samples.shape[0])))
		samples = samples[A,:]
		return samples

def Cat_simulated_annealing(objective, nCats, n_iter):
	# CATEGORICAL SIMULATED_ANNEALING: Function runs simulated annealing
	# algorithm for optimizing functions of categorical inputs.
	#
	# ARGS:
	#  objective: callable, from bounded integer vector-> R
	#  nCats: number of categories, the upper bounds on the integer search space (lower is 0)
	#  n_iter: the number of SA steps to take
	#
	# RETURNS:
	#  model_iter: cumulative argmin
	#  obj_iter: cumulative min


	# Declare vectors to save solutions
	n_vars = len(nCats)
	model_iter = np.zeros((n_iter, n_vars))
	obj_iter   = np.zeros(n_iter)

	# Set initial temperature and cooling schedule
	T = 1.
	cool = lambda T: .8*T

	# Set initial condition and evaluate objective
	old_x   = Cat_sample_inputs(1, nCats)
	old_obj = objective(old_x)

	# Set best_x and best_obj
	best_x   = old_x
	best_obj = old_obj
	t0 = time.time()

	# Run simulated annealing
	for t in range(n_iter):

		# Decrease T according to cooling schedule
		T = cool(T)

		# Find new sample, sample category first, then sample a new element
		new_cat = np.random.randint(n_vars)
		new_cat_i = np.random.randint(nCats[new_cat])
		while new_cat_i is old_x[0, new_cat]:
			nex_cat_i = np.random.randint(nCats[new_cat])

		new_x = old_x.copy()
		new_x[0,new_cat] = new_cat_i

		# Evaluate objective function
		new_obj = objective(new_x)

		# Update current solution iterate
		if (new_obj < old_obj) or (np.random.rand() < np.exp( (old_obj - new_obj)/T )):
			old_x   = new_x
			old_obj = new_obj

		# Update best solution
		if new_obj < best_obj:
			best_x   = new_x
			best_obj = new_obj

		# save solution
		model_iter[t,:] = best_x
		obj_iter[t]		= best_obj

	# print("SA time ", time.time()-t0)

	return (model_iter, obj_iter)




def bhs(Xorg, yorg, burnin, timelimit):
    # Implementation of the Bayesian horseshoe linear regression hierarchy.
    # Parameters:
    #   Xorg     = regressor matrix [n x p]
    #   yorg     = response vector  [n x 1]
    #   nsamples = number of samples for the Gibbs sampler (nsamples > 0)
    #   burnin   = number of burnin (burnin >= 0)
    #   thin     = thinning (thin >= 1)
    #   timelimit= int, seconds of sampling to do before stopping
    #
    # Returns:
    #   beta     = regression parameters  [p x nsamples]
    #   b0       = regression param. for constant [1 x nsamples]
    #   s2       = noise variance sigma^2 [1 x nsamples]
    #   t2       = hypervariance tau^2    [1 x nsamples]
    #   l2       = hypervariance lambda^2 [p x nsamples]
    #
    #
    # Example:
    # % Load a dataset:
    # load hald;
    # % Run horseshoe sampler. Normalising the data is not required.
    # [beta, b0] = bhs(ingredients, heat, 1000, 100, 10);
    # % Plot the samples of the regression coefficients:
    # boxplot(beta', 'labels', {'tricalcium aluminate','tricalcium silicate',...
    #   'tetracalcium aluminoferrite', 'beta-dicalcium silicate'});
    # title('Bayesian linear regression with the horseshoe hierarchy');
    # xlabel('Predictors');
    # ylabel('Beta');
    # grid;
    #
    #
    # References:
    # A simple sampler for the horseshoe estimator
    # E. Makalic and D. F. Schmidt
    # arXiv:1508.03884, 2015
    #
    # The horseshoe estimator for sparse signals
    # C. M. Carvalho, N. G. Polson and J. G. Scott
    # Biometrika, Vol. 97, No. 2, pp. 465--480, 2010
    #
    # (c) Copyright Enes Makalic and Daniel F. Schmidt, 2015
    # Adapted to python by Ricardo Baptista, 2018

    n, p = Xorg.shape

    # Normalize data
    X, _, _, y, muY = standardise(Xorg, yorg)

    # Return values
    # beta = np.zeros((p, ))
    # s2 = np.zeros((1, ))
    # t2 = np.zeros((1, ))
    # l2 = np.zeros((p, ))

    # Initial values
    sigma2  = 1.
    lambda2 = np.random.uniform(size=p)
    tau2    = 1.
    nu      = np.ones(p)
    xi      = 1.

    # pre-compute X'*X (used with fastmvg_rue)
    XtX = np.matmul(X.T,X)

    # Gibbs sampler
    k = 0
    t0 = time.time()
    while(k<burnin and time.time()-t0<timelimit):

        try:
            # Sample from the conditional posterior distribution
            sigma = np.sqrt(sigma2)
            Lambda_star = tau2 * np.diag(lambda2)
            # Determine best sampler for conditional posterior of beta's
            if (p > n) and (p > 200):
                beta = fastmvg(X/sigma, y/sigma, sigma2*Lambda_star)
            else:
                beta = fastmvg_rue(X/sigma, XtX/sigma2, y/sigma, sigma2*Lambda_star)

            # Sample sigma2
            e = y - np.dot(X,beta)
            shape = (n + p) / 2.
            scale = np.dot(e.T,e)/2. + np.sum(beta**2/lambda2)/tau2/2.
            sigma2 = 1. / np.random.gamma(shape, 1./scale)

            # Sample lambda2
            scale = 1./nu + beta**2./2./tau2/sigma2
            lambda2 = 1. / np.random.exponential(1./scale)

            # Sample tau2
            shape = (p + 1.)/2.
            scale = 1./xi + np.sum(beta**2./lambda2)/2./sigma2
            tau2 = 1. / np.random.gamma(shape, 1./scale)

            # Sample nu
            scale = 1. + 1./lambda2
            nu = 1. / np.random.exponential(1./scale)

            # Sample xi
            scale = 1. + 1./tau2
            xi = 1. / np.random.exponential(1./scale)
        except:
            pass

        k = k + 1

    # Re-scale coefficients
    #div_vector = np.vectorize(np.divide)
    #beta = div_vector(beta.T, normX)
    #b0 = muY-np.dot(muX,beta)
    b0 = muY
    # print("bhs time : ", time.time() - t0)
    return (beta, b0, sigma2, tau2, lambda2)

def fastmvg(Phi, alpha, D):
    # Fast sampler for multivariate Gaussian distributions (large p, p > n) of
    #  the form N(mu, S), where
    #       mu = S Phi' y
    #       S  = inv(Phi'Phi + inv(D))
    # Reference:
    #   Fast sampling with Gaussian scale-mixture priors in high-dimensional
    #   regression, A. Bhattacharya, A. Chakraborty and B. K. Mallick
    #   arXiv:1506.04778

    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi,u) + delta
    #w = np.linalg.solve(np.matmul(np.matmul(Phi,D),Phi.T) + np.eye(n), alpha - v)
    #x = u + np.dot(D,np.dot(Phi.T,w))
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:,np.newaxis])
    w = np.linalg.solve(np.matmul(Phi,Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt,w)

    return x

def fastmvg_rue(Phi, PtP, alpha, D):
    # Another sampler for multivariate Gaussians (small p) of the form
    #  N(mu, S), where
    #  mu = S Phi' y
    #  S  = inv(Phi'Phi + inv(D))
    #
    # Here, PtP = Phi'*Phi (X'X is precomputed)
    #
    # Reference:
    #   Rue, H. (2001). Fast sampling of gaussian markov random fields. Journal
    #   of the Royal Statistical Society: Series B (Statistical Methodology)
    #   63, 325-338

    p = Phi.shape[1]
    Dinv = np.diag(1./np.diag(D))

    # regularize PtP + Dinv matrix for small negative eigenvalues
    try:
        L = np.linalg.cholesky(PtP + Dinv)
    except:
        mat  = PtP + Dinv
        Smat = (mat + mat.T)/2.
        maxEig_Smat = np.max(np.linalg.eigvals(Smat))
        L = np.linalg.cholesky(Smat + maxEig_Smat*1e-15*np.eye(Smat.shape[0]))

    v = np.linalg.solve(L, np.dot(Phi.T,alpha))
    m = np.linalg.solve(L.T, v)
    w = np.linalg.solve(L.T, np.random.randn(p))

    x = m + w

    return x

def standardise(X, y):
    # Standardize the covariates to have zero mean and x_i'x_i = 1

    # set params
    n = X.shape[0]
    meanX = np.mean(X, axis=0)
    stdX  = np.std(X, axis=0) * np.sqrt(n)

    # Standardize X's
    #sub_vector = np.vectorize(np.subtract)
    #X = sub_vector(X, meanX)
    #div_vector = np.vectorize(np.divide)
    #X = div_vector(X, stdX)

    # Standardize y's
    meany = np.mean(y)
    y = y - meany

    return (X, meanX, stdX, y, meany)




def cats_to_onehot(x_cat, nCats):
	# categorical to concatenated one hot encoding
	x_cat = x_cat.reshape(-1)

	assert len(x_cat)==len(nCats), "input and nCats must be same length"
	assert all([(0<=xi) & (xi<Ci) for xi, Ci in zip(x_cat, nCats)]), "category is out of range"

	x_bin = [1*(xi==np.arange(Ci)) for xi, Ci in zip(x_cat, nCats)]

	return np.hstack(x_bin)


def onehot_to_cats(x_bin, nCats):
  # concatenated onehot endocings to categorical
	x_bin = x_bin.reshape(-1)
	assert len(x_bin)==np.sum(nCats), "input is longer than number of categories"

	I = np.hstack([0, np.cumsum(nCats)]).astype(int)

	def get_loc(x_bin, l, u):
		return np.where(x_bin[l:u]>0)

	x_cat = [get_loc(x_bin, l, u) for l, u in zip(I[:-1], I[1:])]

	assert all([len(xi)==1 for xi in x_cat]), "input is not one-hot"

	return np.hstack(x_cat)



def check_validity(x, nCats):
    # given a set of integer elements, make sure the set contains at most one from each category
	#
	# ARGS
	#  x: vector of integers between 0 and sum total number of categories
	#  nCats: number of categories for each variable
	#
	# RETURNS
	#  bool: T if x contains at most one category of each variable
	#

	assert 0<=np.min(x) & np.max(x)<np.sum(nCats), "input out of range of categories"

	cumCats = np.hstack([0, np.cumsum(nCats)])
	#partitions = [np.arange(cumCats[i],cumCats[i+1]) for i in range(len(nCats))]
	partitions = [np.arange(i, j) for i, j in zip(cumCats[:-1], cumCats[1:])]
	set_counts = np.array([len(np.intersect1d(x, part_i)) for part_i in partitions])
	if any(set_counts>0) & all(set_counts<=1):
		return True
	else:
		return False



class Cat_LinReg:

	def __init__(self, nCats, order):
		#
		# ARGS
		#  nCats: vector of numbe of categories for each variable
		#  order: the polynomial order of surrogate model
		#
		# RETURNS
		#  NULL
		#

		assert order>0, "order must be a positive integer"
		assert len(nCats)>=order, "higher order than #vars!"

		self.nCats = np.array(nCats).astype(int).reshape(-1,)
		self.order = order
		n_vars     = np.sum(nCats)

		def get_product_terms(ord_i):
			offdProd = np.array(list(combinations(np.arange(n_vars),ord_i)))
			valid = [check_validity(prod_i, self.nCats) for prod_i in  offdProd]
			offdProd = offdProd[valid, :]
			return offdProd

		self.Product_terms = [get_product_terms(i) for i in range(2, order+1)]

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def setupData(self):

		# limit data to unique points
		X, x_idx = np.unique(self.xTrain, axis=0, return_index=True)
		y = self.yTrain[x_idx]

		# set upper threshold
		infT = 1e6

		# separate samples based on Inf output
		y_Infidx  = np.where(np.abs(y) > infT)[0]
		y_nInfidx = np.setdiff1d(np.arange(len(y)), y_Infidx)

		# save samples in two sets of variables
		self.xInf = X[y_Infidx,:]
		self.yInf = y[y_Infidx]

		self.xTrain = X[y_nInfidx,:]
		self.yTrain = y[y_nInfidx]

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def train(self, xTrain_cat, yTrain, timelimit=30):
		# Performs gibbs sampling over coefficients and saves
		# a sample as self.alpha.
		#
		# ARGS
		#  xTrain_cat: matrix of categories/integers, each row is one input
		#  yTrain: vector of doubles, observed values
		#
		# RETURNS
		#  NULL
		#

		# Set nGibbs (Gibbs iterations to run)
		nGibbs = int(1e3)

		# convert x from categorical to full order binary
		self.xTrain = self.order_effects(xTrain_cat, self.order)
		self.yTrain = yTrain

		self.setupData()

		# print("training shapes")
		# print(self.xTrain.shape, self.yTrain.shape)

		(nSamps, nCoeffs) = self.xTrain.shape




		# check if x_train contains columns with zeros or duplicates
		# and find the corresponding indices
		check_zero = np.all(self.xTrain == np.zeros((nSamps,1)), axis=0)
		idx_zero   = np.where(check_zero == True)[0]
		idx_nnzero = np.where(check_zero == False)[0]

		# remove columns of zeros in self.xTrain
		if np.any(check_zero):
			self.xTrain = self.xTrain[:,idx_nnzero]

		# run Gibbs sampler for nGibbs steps
		attempt = 1
		while(attempt):

			# re-run if there is an error during sampling
			# try:
			alphaGibbs,a0,_,_,_ = bhs(self.xTrain,
                                      self.yTrain,
                                      burnin=1000,
                                      timelimit=timelimit)
			# except (KeyboardInterrupt, SystemExit):
			# 	raise
			# except:
			# 	print('error during Gibbs sampling. Trying again.')
			# 	continue

			# run until alpha matrix does not contain any NaNs
			if not np.isnan(alphaGibbs).any():
				attempt = 0

	#	import pdb; pdb.set_trace()

		# append zeros back - note alpha(1,:) is linear intercept
		alpha_pad = np.zeros(nCoeffs)
		alpha_pad[idx_nnzero] = alphaGibbs#[:,-1]
		self.alpha = np.append(a0, alpha_pad)

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def surrogate_model(self, x_cat, alpha):
		# SURROGATE_MODEL: Function evaluates the linear model
		# Assumption: input x only contains one row
		#
		# ARGS
		#  x: 1-row matrix/vector of integers/categories
		#  alpha: lin-reg weight vector
		#
		# RETURNS
		#  x_allpairs: binary matrix, with valid polynomial terms.
		#

		x_cat = x_cat.reshape(-1)
		assert len(x_cat) is len(self.nCats), "surrogate: x_cat is too long/short"
		assert all(x_cat>=0) & all(x_cat<self.nCats), "surrogate: x_cat is out of range"

		x_cat = x_cat.reshape(1,-1)

		# generate x_basis, all binary basis/polynomial vectors
		x_basis = np.append(1, self.order_effects(x_cat, self.order))


		# check if x maps to an Inf output (if so, barrier=Inf)
		barrier = 0.
		if self.xInf.shape[0] != 0:
			if np.equal(x_basis, self.xInf).all(axis=1).any():
				barrier = np.inf


		# compute and return objective with barrier
		out = np.dot(x_basis,alpha) + barrier

		return out

	# ---------------------------------------------------------
	# ---------------------------------------------------------


	def order_effects(self, x_cat, ord_t):
		# Expands integers x_vals into a binary matrix for a linear
		# regression model. Quadratic and higher order terms are filtered
		# for validity.
		#
		# ARGS
		#  x_cat: matrix of integers, each row is one input
		#  ord_t: highest order polynomial terms to include in expansion.
		#
		# RETURNS
		#  x_allpairs: binary matrix, expanded with valid polynomial terms.
		#

		# convert x_cats to one-hot binary vectors x_bin
		x_bin = [cats_to_onehot(xi, self.nCats).reshape(1,-1) for xi in x_cat]
		x_bin = np.vstack(x_bin)
		n_samp, n_vars = x_bin.shape

		# Generate matrix to store results
		x_basis = x_bin

		for i, ord_i in enumerate(range(2, ord_t+1)):
			# generate all VALID combinations of indices
			# offdProd = np.array(list(combinations(np.arange(n_vars),ord_i)))
			# valid = [check_validity(prod_i, self.nCats) for prod_i in  offdProd]
			# offdProd = offdProd[valid, :]
			offdProd = self.Product_terms[i]

			# Compute polynomial terms and augment x values.
			x_comb = np.zeros((n_samp, offdProd.shape[0], ord_i))
			for j in range(ord_i):
				x_comb[:,:,j] = x_bin[:,offdProd[:,j]]
			x_basis = np.append(x_basis, np.prod(x_comb,axis=2),axis=1)

		return x_basis


def plot_progress(Y, scale="linear", ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(np.arange(len(Y)), Y, **kwargs)
    ax.plot(np.arange(len(Y)), np.minimum.accumulate(Y), **kwargs)
    ax.set_xlabel("Number of Observations")

    return(ax)

def plot_2Dpareto(Y, scale="linear", ax=None, **kwargs):
    # Y: N*2 matrix

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    # get pareto optimal points
    dominated = [np.any(np.all(Y<y, axis=1)) for y in Y]
    pareto = np.invert(dominated)
    Y_p = Y[pareto, :]

    # for plotting the pareto front, add ends
    Y_p0 = [Y_p[:,0].min(), Y[:,1].max()]
    Y_p1 = [Y[:,0].max(), Y_p[:,1].min()]
    Y_p  = np.vstack([Y_p0, Y_p, Y_p1])


    Y_p[:,1] = - Y_p[:,1]
    Y_p = np.unique(Y_p, axis=0)
    Y_p[:,1] = - Y_p[:,1]

    ax.scatter(Y[:,0], Y[:,1], **kwargs)
    ax.plot(Y_p[:,0], Y_p[:,1], **kwargs)

    ax.set_xscale(scale)
    ax.set_yscale(scale)

    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_title("Pareto Optimal Points")

    return(ax)
