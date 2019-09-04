import numpy as np

def _get_unique_combinations(idx, r, source_array):

    result = []
    for i in range(idx, len(source_array)):
        if r - 1 > 0:
            #print(idx,r)
            next_level = _get_unique_combinations(i + 1, r - 1, source_array)
            for x in next_level:
                #print('nxt:',next_level)
                value = [source_array[i]]
                #print(value)
                value.extend(x)
                #print('value extended:',value)
                result.append(value)

        else:
            result.append([source_array[i]])

    return result

source=[1,2,3]


_get_unique_combinations(0,2,source)
