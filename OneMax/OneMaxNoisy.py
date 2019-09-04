import numpy as np
import sys

first_arg = sys.argv[1]
V=[]
for letter in first_arg:
    V.append(int(letter))

def OneMax(V):
    ''' Takes V : a vector of vi's (ones or zeros)
    and returns the sum of all (v_i).
    '''
    if __name__ == "__main__":
        return(np.sum(V))

#inputvector = [1,0,1,1,0]

print('result: '+str(OneMax(V)))
