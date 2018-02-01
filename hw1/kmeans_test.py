import mfcc_clustering as mfcc_cls
import configs.hw1_config as config
import svm_classifier as svcclf
import numpy as np
import pickle
import random
train_name_with_labels, trainx, train_name = mfcc_cls.get_mfcc_vecs(config, ['val'])
val_name_with_labels, valx, val_name = mfcc_cls.get_mfcc_vecs(config, ['train'])

model = mfcc_cls.kmeans_training(trainx)
train_y = mfcc_cls.kmeans_test(model, trainx)
val_y = mfcc_cls.kmeans_test(model, valx)

def wrap_svc_data(y, name_with_labels, name, dim = None):
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

    truth_y = [3 if '3' in yy else 2 if '2' in yy else 1 if '1' in yy else 0 for xx, yy in name_with_labels]
    assert(resx.shape[0] == len(truth_y))
    return resx, truth_y, id2name, name2id


trainxx, trainyy, _, _ = wrap_svc_data(train_y, train_name_with_labels, train_name)
print(trainxx.shape, len(trainyy), trainyy[:20])
valxx, valyy, id2name, name2id = wrap_svc_data(val_y, val_name_with_labels, val_name)
import utils
val_all = utils.get_video_and_label_list(config.all_train_list_filename)
val_all = [x for x, y in val_all]
'''
trainxx = np.load('./trainxx.npy')
trainyy = np.load('./trainyy.npy')

valxx = np.load('./valxx.npy')
valyy = np.load('./valyy.npy')
'''
print("TRAINING SVM BEGINS")
clf, scl = svcclf.train(trainxx, trainyy, True)
print("TESTING SVM BEGINS")
y_pred, y_proba = svcclf.test(clf, valxx, scl)
full_y_pred = [0] * len(val_all)
print("TOTAL NUMBER OF VALID VALIDATION DATA %d / %d" %(len(name2id), len(val_all)) )
full_y_proba = np.zeros(shape = (len(val_all), 4))
for i in range(len(val_all)):
    vid_name = val_all[i]
    if vid_name not in name2id:
        full_y_pred[i] = random.randint(0, 4)
        full_y_proba[i] = [0.2, 0.2, 0.2, 0.4]
        continue
    idx = name2id[vid_name]
    full_y_pred[i] = y_pred[idx]
    full_y_proba[i] = y_proba[idx]

np.save('pred', full_y_pred)
for i in [1,2,3]:
    p = full_y_proba[:, i]
    with open('./example_gt_and_pred/P00%d_pred_score.lst'%i, 'w') as f:
        f.write('\n'.join([str(pp) for pp in p]))


