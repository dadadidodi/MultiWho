import os
import sys
sys.path.append("../")
import utils
import numpy as np
import keyframe_extraction
import configs.hw2_config as config 
import hw1.svm_classifier as svmclf

def get_idt_data(config, gps):
    all_video_label_list = []
    if 'train' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_train_list_filename)
    if 'val' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_val_list_filename)
    if 'test' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_test_list_filename)
    idt_full_path = config.idt_bow_full_path
    res = np.zeros(shape=(len(all_video_label_list), 256))
    cnt = 0
    ys = []
    for now_video_label in all_video_label_list:
        vid_name = now_video_label[0]
        ys.append(now_video_label[1])
        fname = os.path.join(idt_full_path, vid_name) + '.npy'
        res[cnt] = np.load(fname)
        cnt += 1
    return res, ys

def convert_y(ys, binary = 0):
    if binary:
        res = [1 if str(binary) in y else 0 for y in ys]
        return res
    res = [1 if '1' in y else 2 if '2' in y else 3 if '3' in y else 0 for y in ys]
    return res 

gps_in = ['train', 'val']
gps_out = ['test'] 

print("Prepare training data")
trainx, trainy = get_idt_data(config, gps_in)
trainy = convert_y(trainy, 0)
print("Prepare testing data")
valx, valy = get_idt_data(config, gps_out)
valy = convert_y(valy, 0)
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
# trainx = np.nan_to_num(trainx)
trainx = np.log(1 + trainx)
# valx = np.nan_to_num(valx)
valx = np.log(1 + valx) 
for ii in range(1, 4):
    trainxx = trainx 
    trainyy = [1 if  ii == yy else 0 for yy in trainy]
    clf, scl = svmclf.train(trainxx, trainyy, C = 10)
    y, y_prob = svmclf.test(clf, valx)
    with open('/home/ubuntu/11775_code/hw1/example_gt_and_pred/P00%d_idt.lst'%ii, 'w') as f:
        f.write('\n'.join([str(x) for x in y_prob[:, 1]]))

