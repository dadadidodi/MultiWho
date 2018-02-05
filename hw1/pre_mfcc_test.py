import mfcc_clustering as mfcc_cls
import configs.hw1_config as config
import svm_classifier as svcclf
import numpy as np
import pickle
import os
import random
train_name_with_labels, trainx, train_name = mfcc_cls.get_mfcc_vecs(config)
val_name_with_labels, valx, val_name = mfcc_cls.get_mfcc_vecs(config, ['val'])
test_name_with_labels, testx, test_name = mfcc_cls.get_mfcc_vecs(config, ['test'])

model = mfcc_cls.kmeans_training(trainx)
train_y = mfcc_cls.kmeans_test(model, trainx)
val_y = mfcc_cls.kmeans_test(model, valx)
test_y = mfcc_cls.kmeans_test(model, testx) 

def pre_wrap_svc_data(y, name_with_labels, name, dim = None):
    pos_mfcc_path = os.path.join(config.dataset_root_path, 'pos_mfcc')
    if not os.path.exists(pos_mfcc_path):
	os.mkdir(pos_mfcc_path)
    if dim is None:
        dim = max(y) + 1
        print("No dimension input. By default, use maximum label %d" % dim)
    assert(len(y) == len(name))
    tot_num = len(set(name))
    resx = np.zeros(shape = (tot_num, dim))
    id2name = {}
    name2id = {}
    for xx, yy in name_with_labels:
        name2id[xx] = len(name2id)
        id2name[len(name2id) - 1] = xx

    for yy, nn in zip(y, name):
        resx[name2id[nn]][yy] += 1

    for k, v in name2id.items():
	vid_name = k
        vid = v
	np.save(os.path.join(pos_mfcc_path, vid_name +'.npy'), resx[vid].reshape((1, dim))) 
		
pre_wrap_svc_data(train_y, train_name_with_labels, train_name, dim = 200)
pre_wrap_svc_data(val_y, val_name_with_labels, val_name, dim = 200)
pre_wrap_svc_data(test_y,test_name_with_labels, test_name, dim = 200)

