import sys
sys.path.append("../")
import utils 
import threading
import numpy as np
import configs.hw2_config as config
import hw1.svm_classifier as svmclf
import os
def get_cnn_fts(config, gps = ['train']):
    all_video_label_list = []
    if 'train' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_train_list_filename)
    if 'val' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_val_list_filename)
    if 'test' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_test_list_filename)
    print("get_surf_kps %d"%len(all_video_label_list))
    
    cnn_fts_path = os.path.join(config.dataset_root_path, "vgg16_fts")
    fnames = os.listdir(cnn_fts_path)
    xs = []
    ys = []
    vids = []
    for now_video_label in all_video_label_list:
        vid_name = now_video_label[0]
        label = now_video_label[1]

        tmp_name = vid_name +'_'
        files = [fname for fname in fnames if tmp_name in fname]
        for fname in files[:200]:
            npy = np.load(os.path.join(cnn_fts_path, fname)).reshape(-1)[0::6]
            if npy is None:
                continue
            xs.append(npy)
            ys.append(label)
            vids.append(vid_name)
        print("FINISH %d"%(len(ys) / 100))
    return np.array(xs), ys, vids 

def get_cnn_res(config, clf, gps = ['val']):
    all_video_label_list = []
    if 'train' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_train_list_filename)
    if 'val' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_val_list_filename)
    if 'test' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_test_list_filename)
    print("get_surf_kps %d"%len(all_video_label_list))
    
    cnn_fts_path = os.path.join(config.dataset_root_path, "vgg16_fts")
    fnames = os.listdir(cnn_fts_path)
    xs = []
    ys = []
    for now_video_label in all_video_label_list:
        vid_name = now_video_label[0]
        label = now_video_label[1]
        tmp_name = vid_name +'_'
        print(vid_name)
        files = [fname for fname in fnames if tmp_name in fname]
        for fname in files[:200]:
            npy = np.load(os.path.join(cnn_fts_path, fname)).reshape(-1)[0::6]
            if npy is None:
                continue
            xs.append(npy)
            y, y_prob =  svmclf.test(clf, np.array(xs))
            y_prob = np.mean(y_prob, axis = 1)
        print("FINISH %d"%(len(xs) / 100))
    return np.array(ys) 

def get_cnn_model(config, gps_in):
    xs, ys = get_cnn_fts(config, gps_in)



