import numpy as np
import subprocess
import matplotlib.pyplot as plt
from BOCS import Cat_BOCS1#, MO_Cat_BOCS


#########################################################################
########################### SINGLE OBJECTIVE ############################
#########################################################################

print("\nMinimizing OneMax function, x_max=[1,1,1,1,1,1,1,1,1], y_max= 10 \n")

np.random.seed(20)

# Give the number of categories of each element
n=20
n_Cats = np.array(2*np.ones(n),dtype=int) #[2,2,2,2,2,2]
print(n_Cats)
# define objective
def objective(x):
    xx=str()
    for i in x:
        xx+=str(i)
    cmd=['python', '/home/pkent/Documents/Solo Research Project/OneMax/OneMax.py',xx]
    output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
#print(output.decode('utf-8'))
#print(output.decode('utf-8')[8:])
    return -int(output.decode('utf-8')[8:])
    # x = x.reshape(9,)
    # return -np.sum(x)

# call the optimizer



if __name__ == "__main__":
    ## How many runs to compare
    runs =4
    ## How many iterations per run
    iterations = 60
    ARRAY=np.zeros((runs,iterations))
    TimerArray=np.zeros((runs,iterations-2))
    for i in range(runs):
        ARRAY[i,:],TimerArray[i,:]=Cat_BOCS1(f=objective, n_Cats=n_Cats, n_init=np.max(n_Cats), n_evals=iterations,verbose=True,knownmax=n)
    plt.errorbar(range(1,iterations+1),ARRAY.mean(axis=0),xerr=0,yerr=ARRAY.std(axis=0))
    plt.title('BOCs running on MaxOne Function with n=%i over %i runs' %(iterations,runs))
    plt.xlabel('runs')
    plt.ylabel('fit')
    plt.ylim((0,n))
    print('Wall Clock time: ', np.mean(np.sum(TimerArray,axis=1)),'max: ', np.max(np.sum(TimerArray,axis=1)),'min: ',np.min(np.sum(TimerArray,axis=1)))
    plt.show()
