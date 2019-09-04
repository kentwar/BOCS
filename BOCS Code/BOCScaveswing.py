import numpy as np
import subprocess
import matplotlib.pyplot as plt
from BOCS import Cat_BOCS, MO_Cat_BOCS

#########################################################################
########################### SINGLE OBJECTIVE ############################
#########################################################################

print("\nMinimizing CaveSwing function, \n")

np.random.seed(20)

##Give the number of categories of each element
n=5
n_Cats = [7,2,2,4,3]
print(n_Cats)
# define objective
def objective(x):
    xx=str()
    for i in x:
        xx+=str(i)
    cmd=['/usr/lib/jvm/java-12-oracle/bin/java', '-Dval1=I can pass Variables to the java code!', '-Dfile.encoding=UTF-8', '-classpath', '/home/pkent/Documents/Solo Research Project/SimpleAsteroids/src/', 'caveswing.test.EvoAgentVisTest', xx]
    output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
#print(output.decode('utf-8'))
#print(output.decode('utf-8')[8:])
    return -int(output.decode('utf-8'))




# call the optimizer

if __name__ == "__main__":
    ## How many runs to compare
    runs =3
    ## How many iterations per run
    iterations = 20
    ARRAY=np.zeros((runs,iterations))
    for i in range(runs):
        ARRAY[i,:]=Cat_BOCS(f=objective, n_Cats=n_Cats, n_init=np.max(n_Cats), n_evals=iterations, verbose=True)
    plt.errorbar(range(1,iterations+1),ARRAY.mean(axis=0),xerr=0,yerr=ARRAY.std(axis=0))
    plt.title('BOCs running on MaxOne Function with n=%i over %i runs' %(iterations,runs))
    plt.xlabel('runs')
    plt.ylabel('fit')
    plt.ylim((0,n))
    plt.show()




# cmd=['/usr/lib/jvm/java-12-oracle/bin/java', '-Dval1=I can pass Variables to the java code!', '-Dfile.encoding=UTF-8', '-classpath', '/home/pkent/Documents/Solo Research Project/SimpleAsteroids/src/', 'caveswing.test.EvoAgentVisTest', '50031']
# output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
# print(output.decode('utf-8'))
