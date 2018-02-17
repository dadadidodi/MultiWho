import os
import sys
sys.path.append("../")
import utils 
import pickle 
import numpy as np
import configs.hw1_config as config
import svm_classifier as svcclf

def wrap_asr_data(config, tags = ['train']):
    all_video_label_list = []
    if 'train' in tags:
	all_video_label_list += utils.get_video_and_label_list(config.all_train_list_filename)
    if 'val' in tags:
	all_video_label_list += utils.get_video_and_label_list(config.all_val_list_filename)
    if 'test' in tags:
	all_video_label_list += utils.get_video_and_label_list(config.all_test_list_filename)
    print("TOTAL NUMBER OF DOCS %d "%len(all_video_label_list))
    root_path = config.asr_bow_root_path
    vid_names = set([item[0] for item in all_video_label_list])
    all_files = os.listdir(root_path)
    x = np.zeros(shape=(len(all_video_label_list), 3000))
    y = []
    cnt1 = 0; cnt2=0; cnt3=0
    for now_video_label in all_video_label_list:
	vid_name = now_video_label[0]
        label = 0
        if '1' in now_video_label[1]:
	    label = 1
	    cnt1 +=1 
    	if '2'  in now_video_label[1]:
	    label = 2
     	    cnt2 += 1
        if '3'  in now_video_label[1]:
	    label = 3
            cnt3+=1
    	subx = np.load(os.path.join(root_path, vid_name+'.npy'))
        x[len(y), :] = subx
        y.append(label)
    print(x.shape, len(y), cnt1, cnt2, cnt3)
    return x, y 

trainx, trainy = wrap_asr_data(config, ['train', 'val']) 
valx, valy = wrap_asr_data(config, ['test'])

print("TRAINING SVM BEGINS")
#########################One model for all########################
'''
clf, scl = svcclf.train(trainx, trainy, True) 
y_pred, y_proba = svcclf.test(clf, valx, scl)

for i in [1,2,3]:
    with open('./example_gt_and_pred/P00%d_pred_score.lst'%i, 'w') as f:
	f.write('\n'.join([str(pp) for pp in y_proba[:, i]]))
'''
##########################One model for one class#################
for i in [1, 2, 3]:
    trainyy = [1 if x == i else 0 for x in trainy]
    clf, scl = svcclf.train(trainx, trainyy, True, C = 10000)
    _, y_proba = svcclf.test(clf, valx, scl)

    p = y_proba[:, 1]
    with open('./example_gt_and_pred/asr_P00%d_pred_score.lst' % i, 'w') as f:
	f.write('\n'.join([str(pp) for pp in p]))


