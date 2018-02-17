import os
import sys
sys.path.append("../")
import utils
import numpy as np
import keyframe_extraction
import configs.hw1_config as config 
import hw1.svm_classifier as svmclf

gps_in = ['train']
gps_out = ['train'] 

model = keyframe_extraction.get_surf_knn_model(config, gps_in)
print("Prepare training data")
trainx, trainy = keyframe_extraction.surf_histogram_builder(config, model, gps_out)
print("Prepare testing data")
valx, valy = keyframe_extraction.surf_histogram_builder(config, model, ['test']) 
print(trainy)

'''
clf, scl = svmclf.train(trainx, trainy)
print("--------------FINISH TRAINING")
print(valx.shape)
y, y_pred = svmclf.test(clf, valx)
for i in [1,2,3]:
    ys = y_pred[:, i]
    with open('/home/ubuntu/11775_code/hw1/example_gt_and_pred/P00%d_pred_score.lst'%i, 'w') as f:
        f.write('\n'.join([str(x) for x in ys]))
'''
trainx = np.nan_to_num(trainx)
trainx = np.log(1 + trainx)
np.save('trainx', trainx)
np.save('trainy', trainy)

valx = np.nan_to_num(valx)
valx = np.log(1 + valx)
np.save('valx', valx)
np.save('valy', valy)

# valx = np.log(1 + valx) 
for ii in range(1, 4):
    trainxx = trainx 
    trainyy = [1 if str(ii) in yy else 0 for yy in trainy]
    clf, scl = svmclf.train(trainxx, trainyy, C = 100000)
    y, y_prob = svmclf.test(clf, valx)
    with open('/home/ubuntu/11775_code/hw1/example_gt_and_pred/P00%d_surf.lst'%ii, 'w') as f:
        f.write('\n'.join([str(x) for x in y_prob[:, 1]]))

