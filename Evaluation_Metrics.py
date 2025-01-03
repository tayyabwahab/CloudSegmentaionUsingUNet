import numpy as np


accuracies = []
temp = 0
print(len(predicts))
for i in range(len(predicts)):
  compare = np.equal(predicts[i], GTMasks[i])
  if abs(np.average(predicts[i]) - np.average(GTMasks[i])) < 60:
    temp = temp+1
  accuracy = np.sum(compare)
  acc = accuracy/len(predicts[i].flatten())
  accuracies.append(acc)
acca = temp/len(predicts)*100
final_accuracy = np.sum(accuracies)/len(accuracies)
print(final_accuracy)
print(acca)

all_tp = []
all_fp = []
all_fn = []

for k in range(len(predicts)):
  tp = 0
  fp = 0
  fn = 0
  for i in range(300):
    for j in range(300): 
      if predicts[k][i][j] == GTMasks[k][i][j]: 
        if predicts[k][i][j] == 255:
          tp = tp+1

      if predicts[k][i][j] != GTMasks[k][i][j]:
        if predicts[k][i][j] == 255:
          fp = fp+1
        else: 
          fn = fn+1
  all_tp.append(tp)
  all_fp.append(fp)
  all_fn.append(fn)

precision = []
recall = []
fscore = []

for i in range(len(all_tp)):
  if all_tp[i] == 0 and all_fp[i] == 0:
    continue
  p = all_tp[i]/(all_tp[i]+all_fp[i])
  r = all_tp[i]/(all_tp[i]+all_fn[i])
  precision.append(p)
  recall.append(r)
  if p ==0 and r == 0:
    print(i)
    continue
  f = 2*p*r/(p+r)
  fscore.append(f)
