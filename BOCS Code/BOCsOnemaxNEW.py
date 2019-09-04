import numpy as np
import subprocess
import datetime
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import time
from BOCS import Cat_BOCS1#, MO_Cat_BOCS


#########################################################################
########################### SINGLE OBJECTIVE ############################
#########################################################################

print("\nMinimizing Noisy Trap function, x_max=[1,1,1,1,1,1,1,1,0], y_max= 9 \n")

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

def noisyobjective(x):
    xx=str()
    for i in x:
        xx+=str(i)
    cmd=['python', '/home/pkent/Documents/Solo Research Project/OneMax/OneMax.py',xx]
    output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
#print(output.decode('utf-8'))
#print(output.decode('utf-8')[8:])
    #noise = np.random.normal(0,1,1)
    returnval = -int(output.decode('utf-8')[8:])
    return returnval#+noise
    # x = x.reshape(9,)
    # return -np.sum(x)
# call the optimizer

def savetofile(array,directory,filename):
    '''A functon takes np.array:array and String:directory and saves it in the directory'''
    if not os.path.exists(directory):
        os.makedirs(directory)
    timeval = datetime.datetime.now().strftime("%c")
    np.savetxt(directory+"/"+timeval+filename+".csv", array, delimiter=",")

if __name__ == "__main__":
    ## How many runs to compare
    runs =100
    ## How many iterations per run
    iterations = 80
    ARRAY=np.zeros((runs,iterations))
    TrueArray=np.zeros((runs,iterations))
    TimerArray=np.zeros((runs,iterations-2))
    for i in range(runs):
        ARRAY[i,:],TimerArray[i,:],TrueArray[i,:]=Cat_BOCS1(fnoisy=noisyobjective,f=objective, n_Cats=n_Cats, n_init=np.max(n_Cats), n_evals=iterations,verbose=True,knownmax=0)
    #plt.errorbar(range(1,iterations+1),ARRAY.mean(axis=0),xerr=0,yerr=ARRAY.std(axis=0))
    #Convert arrays to dataframes
    FitnessArray2=pd.DataFrame(TrueArray)
    TimeArray2=pd.DataFrame(TimerArray)
    #Export as csv
    FitnessArray2.to_csv(('Data/runs:'+str(runs)+'it:'+str(iterations)+'n:'+str(n)+'  '+str(time.strftime("%d %b %H:%M:%S",time.gmtime()))+'BOCS_Fitness.csv'))
    TimeArray2.to_csv(('Data/runs:'+str(runs)+'it:'+str(iterations)+'n:'+str(n)+'  '+str(time.strftime("%d %b %H:%M:%S",time.gmtime()))+'BOCS_Timer.csv'))

    plt.errorbar(range(1,iterations+1),TrueArray.mean(axis=0),xerr=0,yerr=ARRAY.std(axis=0),label='True Value')
    plt.plot(range(3,iterations+1),TimerArray.mean(axis=0),label='time per iteration')
    plt.title('BOCs running on MaxOne Function with n=%i over %i runs' %(iterations,runs))
    plt.legend()
    plt.xlabel('runs')
    plt.ylabel('fit')
    plt.ylim((0,n))
    print('Wall Clock time: ', np.mean(np.sum(TimerArray,axis=1))/np.max(np.sum(TimerArray,axis=1))*n,'max: ', np.max(np.sum(TimerArray,axis=1)),'min: ',np.min(np.sum(TimerArray,axis=1)))
    savetofile(TimerArray,'/home/pkent/Documents/Solo Research Project/BOCS Code/Output','Time, n: '+str(n)+' iter: '+str(iterations)+' ')
    savetofile(TrueArray,'/home/pkent/Documents/Solo Research Project/BOCS Code/Output','True, n: '+str(n)+' iter: '+str(iterations)+' ')
    plt.show()
