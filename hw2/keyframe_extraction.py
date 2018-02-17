import os
import sys
sys.path.append("../")
import utils
import threading
import numpy as np 
import cv2
max_th=5
np.random.seed(0)
from hw1.mfcc_clustering import kmeans_training, kmeans_test
def cmd_runner(cmd):
    os.system(cmd)

def image_extraction(config):
    if not os.path.exists(config.ds_video_root_path):
        os.mkdir(config.ds_video_root_path)

    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)

    thread_pool=[]
    cnt = 0
    skip = 0
    for now_video_label in all_video_label_list:
        cnt += 1
        vid_name=now_video_label[0]
        vid_filename=os.path.join(config.video_root_path,vid_name+config.video_file_format)
        ds_vid_filename=os.path.join(config.ds_video_root_path,vid_name+config.video_file_format)
        keyframe_filename=ds_vid_filename.replace('down_samp_video', 'surf_images')
        keyframe_filename=keyframe_filename.replace('.mp4', '_\%04d.jpg')
        if not os.path.isfile(ds_vid_filename):
            skip += 1
            continue
        
        print "Extract key frame for video : ",vid_filename, " to ", keyframe_filename

        ffmpeg_cmd="ffmpeg -ss 0 -i %s -vf fps=3 -loglevel error %s -hide_banner"%(ds_vid_filename, keyframe_filename)
        # print ffmpeg_cmd

        while len(threading.enumerate())>=max_th:
            pass

        now_th=threading.Thread(target=cmd_runner,args=[ffmpeg_cmd])
        now_th.start()
        thread_pool.append(now_th)
        print("Finish %d files Skip %d files " %(cnt - skip, skip))
    for th in thread_pool:
        th.join()

# TODO
def surf_histogram_builder(config, model, gps_out):
    all_video_label_list = []
    if 'train' in gps_out:
        all_video_label_list += utils.get_video_and_label_list(config.all_train_list_filename)
    if 'val' in gps_out:
        all_video_label_list += utils.get_video_and_label_list(config.all_val_list_filename)
    if 'test' in gps_out:
        all_video_label_list += utils.get_video_and_label_list(config.all_test_list_filename)

    surf_kps_root_path = os.path.join(config.dataset_root_path, "surf_feature")
    print("READ FROM %s %s %d"%(surf_kps_root_path,'_'.join(gps_out), len(all_video_label_list) ))
    fnames = os.listdir(surf_kps_root_path)
    histx= np.empty(shape = (len(all_video_label_list), model.n_clusters * 2))
    histy = []
    cnt = 0
    for now_video_label in all_video_label_list:
        prefix = now_video_label[0] + '_'
        matched_fnames = [x for x in fnames if prefix in x]
        tmpx = []
        tmpy = []
        histy.append(now_video_label[1])
        hist_tmp = np.empty(shape = (len(matched_fnames), model.n_clusters)) 
        cnt_frame = 0
        for fname in matched_fnames:
            tmpxx = np.load(os.path.join(surf_kps_root_path, fname))
            tmpx.append(tmpxx)
            if len(tmpxx) > 0: 
                tmpyy = kmeans_test(model, tmpxx)
            for yy in tmpyy:
                hist_tmp[cnt_frame][yy] += 1
            cnt_frame+=1
        if len(tmpx) > 0:
            tmpx = np.concatenate(tmpx, axis = 0)
        if hist_tmp is None or len(hist_tmp) == 0:
            cnt += 1
            continue
        max_pool = np.max(hist_tmp, axis = 0)
        avg_pool = np.mean(hist_tmp, axis = 0)
        histx[cnt] = np.concatenate([max_pool, avg_pool], axis = 0)
        print("deal with %dth file "%cnt )
        cnt += 1
    return histx, histy 
  
def get_surf_kps(config, gps = ['train']): 
    all_video_label_list = []
    if 'train' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_train_list_filename)
    if 'val' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_val_list_filename)
    if 'test' in gps:
        all_video_label_list += utils.get_video_and_label_list(config.all_test_list_filename)
    print("get_surf_kps %d"%len(all_video_label_list))

    surf_kps_root_path = os.path.join(config.dataset_root_path, "surf_feature")
    print("surf_kps_root_path %s"%surf_kps_root_path)
    video_set = set([x[0] + '_' for x in all_video_label_list])
    fnames = sorted(os.listdir(surf_kps_root_path))
    surf_kps_num = 0
    npy = None
    rnd = 0 
    npyshape=None
    for fname in fnames:
        if fname[:-8] in video_set: 
            npy = np.load(os.path.join(surf_kps_root_path, fname))
            if npy is None or len(npy.shape) != 2:
                continue
            # npy = npy[rnd::100]
            # rnd += (rnd + 1) % 100
            surf_kps_num += npy.shape[0]
            npyshape=npy.shape
            del npy
    print("NPY shape should be ", npyshape, surf_kps_num)
    surf_kps = np.empty(shape=(surf_kps_num, npyshape[1]))
    
    st = 0
    for fname in fnames:
        if fname[:-8] in video_set:
            npy = np.load(os.path.join(surf_kps_root_path, fname))
            if npy is None:
                continue
            en = st + npy.shape[0]
            surf_kps[st:en] = npy
            del npy
    return surf_kps

def get_surf_knn_model(config, gps = ['train']):
    trainx = get_surf_kps(config, gps)
    print("knn training ", trainx.shape, np.sum(trainx))
    model = kmeans_training(trainx)
    return model

def surf_extraction(config):
    key_frame_root_path = config.ds_video_root_path.replace('down_samp_video', 'surf_images')
    fnames = os.listdir(key_frame_root_path)
    cnt = 0 
    for fname in fnames:
        cnt += 1
        key_frame_path = os.path.join(key_frame_root_path, fname)
        img = cv2.imread(key_frame_path, 0)
        if img is None:
            continue
        # Surf extraction
        surf =cv2.SURF(4000)
        _, descriptors = surf.detectAndCompute(img, None)
        if descriptors is None:
            continue
        outpath = key_frame_path.replace('surf_images', 'surf_feature')
        outpath = outpath[:-4]
        # Save surf features as np array 
        print(descriptors.shape)
        # np.save(outpath, descriptors)
        if cnt % 100 == 0:
            print("Finished %d"%int(100.0 * cnt / len(fnames)))

if __name__== "__main__":
    pass
    import configs.hw2_config as config
    # image_extraction(config)
    # surf_extraction(config)
