import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

X1=np.mat([[2,1],[1,5]])
kernel(X1,X1)
kernel(X1,X1)

X1= [[2,1,4],[1,5,4],[3,1,4]]
X2= [[1,2,5],[1,5,1],[2,1,0]]
X2= np.mat([[3,4],[2,5]])

kernel(X1,X2,1)
RBF_Kernel(X,X)

def RBF_Kernel(X1,X2,l=1.0,sigma_f=1.0):
    ''' Isotropic squared exponential kernel. Computes a covariance
    matrix from points in X1 and X2. Args: X1: Array of m points (m x d).
    X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    x1norm=[(v*v.T)[0,0] for v in X1]
    x2norm=[(v*v.T)[0,0] for v in X2]
    x1squared = x1norm*np.ones((X2.shape[0],len(x1norm)))
    x2squared = np.mat(x2norm).T
    sqdist=(x1squared+x2squared-2*np.dot(X1,X2.T).T).T
    return sigma_f**2 * np.exp(-(0.5* l**2) * sqdist)


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = np.asarray(X).ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)

# Finite number of points
X = np.mat(np.arange(-5, 5, 0.2).reshape(-1, 1))
X2 = np.arange(-5, 5, 0.2).reshape(-1, 1)
X1=np.arange(-4, 5, 0.2).reshape(-1, 1)
# Mean and covariance of the prior
mu = np.zeros(X.shape)
cov = RBF_Kernel(X, X)

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

# Plot GP mean, confidence interval and samples
plot_gp(mu, cov, X, samples=samples)

plt.show()

from numpy.linalg import inv

def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    ''' Computes the sufficient statistics of the GP posterior predictive distribution from m training data X_train and Y_train and n new inputs X_s. Args: X_s: New input locations (n x d). X_train: Training locations (m x d). Y_train: Training targets (m x 1). l: Kernel length parameter. sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
    K = RBF_Kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(X_train.shape[0])
    K_s = RBF_Kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(X_s.shape[0])
    K_inv = inv(K)

    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

(RBF_Kernel(X, X, 1, 1) + (1e-8)**2 * np.eye(X.shape[0])).shape

# Noise free training data
X_train = np.mat(np.array([-4, -3, -2, -1, 1]).reshape(-1, 1))
Y_train = np.mat(np.sin(X_train))

# Compute mean and covariance of the posterior predictive distribution
mu_s, cov_s = posterior_predictive(X, X_train, Y_train)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)

plot_gp(mu_s, cov_s, X.A1, X_train=X_train, Y_train=Y_train, samples=samples)

from sklearn import gaussian_process

kernel(np.array(X2),np.array(X1))
RBF_Kernel(np.mat(X2),np.mat(X1))
gaussian_process.kernels.RBF()(X2,X1)
metrics.pairwise.rbf_kernel()(X2,X1)


def RBF_Kernel(X1,X2,l=1,sigma=1):
    '''This function returns the Squared Exponential kernel for
    use in Gaussian Process regression'''
    X1norm=[np.dot(v,np.transpose(v))[0,0] for v in X1]
    X2norm=np.mat([np.dot(v,np.transpose(v))[0,0] for v in X2])
    X1_Squared=(X1norm*np.ones((X2.shape[0],1))).T
    X2_Squared=np.mat(X2norm).T
    X1X2=X1*X2.T
    squaredval=X1_Squared+X2_Squared.T-2*(X1*X2.T)
    return(np.exp(-(1.0 / 2)*(squaredval/l**2)))
