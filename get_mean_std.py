from statistics import mean
import numpy as np

file_name = input() + '.out'
ll = len('Best test accuracy: ')
lll = len('Best test accuracy: 0.0000 at iter. ')

acc = []

with open(file_name, 'r') as f:
    for content in f.readlines():
        if 'Best test accuracy: ' in content and float(content[lll:]) > 0:
            acc.append(float(content[ll:ll+6]))

print(acc)
print(mean(acc)*100, np.std(acc)*100)
