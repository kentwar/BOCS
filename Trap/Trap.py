import numpy as np
import sys

first_arg = sys.argv[1]
V=[]
for letter in first_arg:
    V.append(int(letter))

def Trap(V):
    ''' Takes V : a vector of vi's (ones or zeros)
    and returns the sum of all (v_i), However a vector of all ones =0
    '''
    if __name__ == "__main__":
        value = np.sum(V)
        if value<len(V):
            return(value)
        else:
            return(0)

#inputvector = [1,0,1,1,0]

print('result: '+str(Trap(V)))
