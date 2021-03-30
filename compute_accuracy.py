import torch
import numpy
 
gt_targets = torch.load('scan_gt_targets.pth.tar')
cluster_preds = torch.load('scan_cluster_predictions.pth.tar')

print(gt_targets.numpy()[0])

with open('scan_gt_targets.txt', 'w') as writeFile:
  for targ in gt_targets.numpy():
    writeFile.write(str(targ) + '\n')

with open('scan_cluster_predictions.txt', 'w') as writeFile:
  for pred in cluster_preds.numpy():
    writeFile.write(str(pred) + '\n')

