import numpy as np
from BOCS import Cat_BOCS, MO_Cat_BOCS


#########################################################################
########################### SINGLE OBJECTIVE ############################
#########################################################################

print("\nMinimizing toy function, 3D bowl, x_min=[3,3,3], y_min=0 \n")

np.random.seed(1)

# Give the number of categories of each element
n_Cats = np.array([5, 6, 4])

# define objective
def objective(x):
    x = x.reshape(3,)
    return np.sum((x - 3.)*(x-3.))

# call the optimizer
Cat_BOCS(f=objective, n_Cats=n_Cats, n_init=6, n_evals=10, verbose=True)



