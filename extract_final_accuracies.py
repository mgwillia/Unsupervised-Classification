import numpy as np

accs = []
with open('scan_cub_repeat.out.751823', 'r') as readFile:
    lines = readFile.readlines()
    for line in lines:
        if 'Final Accuracy' in line:
            print(line)
            acc = float(line.strip().split(' ')[2])
            accs.append(acc)
print(np.mean(accs))

accs = []
with open('scanc_cub_repeat.out.751822', 'r') as readFile:
    lines = readFile.readlines()
    for line in lines:
        if 'Final Accuracy' in line:
            print(line)
            acc = float(line.strip().split(' ')[2])
            accs.append(acc)
print(np.mean(accs))
