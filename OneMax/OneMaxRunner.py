import subprocess
import time

Results = []

def runtest(V):
    cmd=['python', '//home/pkent/Documents/Solo Research Project/OneMax/OneMax.py',V]
    output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
    print(output.decode('utf-8'))
    print(output.decode('utf-8')[8:])
    Results.append(output.decode('utf-8')[8:])

runtest('001001')
