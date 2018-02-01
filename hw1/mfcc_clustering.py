import os
import sys
sys.path.append("../")
import utils
import pickle
import numpy as np
import kmc2
<<<<<<< HEAD
from sklearn.cluster import MiniBatchKMeans, KMeans

def get_mfcc_vecs(config, gps = ['train'], stride = 5):
=======
from sklearn.cluster import MiniBatchKMeans

def get_mfcc_vecs(config, gps = ['train'], stride = 3):
>>>>>>> a9c8cedab10dae3829563c37918a3aeaeb4d40bf
    #   read and concatenate train/validation/test video lists
    print("GET NUMPY MFCC FEATURES FOR %s with stride %d" % (' and '.join(gps), stride) )
    print("file path %s"% config.all_train_list_filename)
    all_video_label_list = []
    if 'train' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_train_list_filename)
    if 'val' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_val_list_filename)
    if 'test' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_test_list_filename)

    vid_names = set([item[0] for item in all_video_label_list])
    all_files = os.listdir(config.mfcc_root_path)
    # Only include files that appear in numpy files
    all_files = [item for item in all_files if item[:-8] in vid_names]
    valid_audio_names = set([item[:-8] for item in all_files if item[:-8] in vid_names])
    valid_video_label_list = [x for x in all_video_label_list if x[0] in valid_audio_names]
    print("........TOTAL NUMBER OF INTERESTED VIDEOS %d" % len(vid_names))
    print("........TOTAL NUMBER OF NUMPY FILES NEEDED TO BE INTERESTED IN %d" % len(all_files))

    print("........TOTAL NUMBER OF AUDIO FILES NEEDED TO BE INTERESTED IN %d" % len(valid_audio_names))
    print("........%d " % (len(valid_video_label_list)))
    mfcc_vec_num = 0
    for mfcc_part_fn in all_files:
        npy = np.load(os.path.join(config.mfcc_root_path, mfcc_part_fn))
        mfcc_vec_num += int((npy.shape[1] + stride - 1) / stride)
<<<<<<< HEAD
    mfcc_vecs = np.empty(shape = (mfcc_vec_num, 13))
=======
    mfcc_vecs = np.empty(shape = (mfcc_vec_num, 20))
>>>>>>> a9c8cedab10dae3829563c37918a3aeaeb4d40bf
    print("........TOTAL MFCC VEC NUMS: %d"%mfcc_vecs.shape[0])

    st = 0

    vec_vid_names = [None] * mfcc_vecs.shape[0]
    for mfcc_part_fn in all_files:
        npy = np.load(os.path.join(config.mfcc_root_path, mfcc_part_fn))[:, 0::stride]
        en = st + npy.shape[1]
        vec_vid_names[st:en] = [mfcc_part_fn[:-8]] * (en - st)
        mfcc_vecs[st:en, :] = npy.transpose()
        st = en

    print("----------------------------------")
    np.save(os.path.join(config.mfcc_root_path, '_'.join(gps)), mfcc_vecs)

    return valid_video_label_list, mfcc_vecs, vec_vid_names

def kmeans_training(X, seed_num = 200):
    print('KMEANS TRAINING WITH %d DATAPOINTS EACH WITH DIMENSION %d'%(X.shape[0], X.shape[1]) )
    print("........KMEANS TRAINING BEGINS...............")
<<<<<<< HEAD
    
=======
>>>>>>> a9c8cedab10dae3829563c37918a3aeaeb4d40bf
    seeding = kmc2.kmc2(X, seed_num, random_state = 11)
    print("........SEEDING PROCESSED THROUGHT KMCMC")

    model = MiniBatchKMeans(seed_num, init = seeding, random_state = 23).fit(X)
<<<<<<< HEAD
    
    # model = KMeans(seed_num, random_state = 23).fit(X)
=======
>>>>>>> a9c8cedab10dae3829563c37918a3aeaeb4d40bf
    return model

def kmeans_test(model, X):
    if model is None:
        print("Model needed to compute cluster belongings")
    y = model.predict(X)
    return y

if __name__=="__main__":
    pass
