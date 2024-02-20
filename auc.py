import subprocess
import os


file_train = 'english.train'
file_test1 = 'english.test'
file_test2 = 'tagalog.test'

n = 10
r = 4

file_test = file_test1
cmd = f'java -jar negsel2.jar -self {file_train} -n {n} -r {r} -c -l < {file_test}'

p = os.popen(cmd)
print(p.read())
