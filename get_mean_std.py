import numpy as np

file_name = input() + '.out'
ll = len('Best test accuracy: ')
lll = len('Best test accuracy: 0.0000 at iter. ')

if 'COVID' in file_name:
    th = 0.5
elif 'Came' in file_name:
    th = 0.5
elif 'kvasir' in file_name:
    th = 0.33

acc = []

with open(file_name, 'r') as f:
    for content in f.readlines():
        if 'Best test accuracy: ' in content and float(content[lll:]) > -1:
            a = float(content[ll:ll+6])
            if a > th:
                acc.append(a)

# remove outliers
acc = np.array(acc)
mean = np.mean(acc)
std_dev = np.std(acc)
z_scores = (acc - mean) / std_dev

acc_new = []
for a, z in zip(acc, z_scores):
    if abs(z) < 1:
        acc_new.append(a.item())

print(acc)
print(acc_new)
if len(acc_new) <= 1:
    print(np.mean(acc)*100, np.nan)
else:
    print(np.mean(acc_new)*100, np.std(acc_new)*100)
