"""
    A reference code for improved dense trajectories feature extraction and encoding on MED videos...
"""

import sys
sys.path.append("../")
sys.path.append("./")
import os
import utils
import threading
import gzip
import numpy as np
import time
max_th=6

# number of bag-of-words centers...
num_centers=256
def cmd_runner(cmd):
    os.system(cmd)

def feat_ext(config, exceptions):
    avi_root_path = config.avi_video_root_path
    idt_raw_feat_root_path=config.idt_raw_root_path
    if not os.path.exists(idt_raw_feat_root_path):
        os.mkdir(idt_raw_feat_root_path)

    st = time.time() 

    # get the combined training and testing video list
    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)

    #TODO: extracting improved DenseTrajectory feature for med videos

    # Given video's filename: video_filename (in .avi format) and the improved dense trajectory feature filename: vid_idt_filename,
    # using the following command for iDT feature extraction...
    # Think about how to accelerate the feature extraction process ? 
    thread_pool = []
    for now_video_label in all_video_label_list:
        vid_name = now_video_label[0]
        if vid_name in exceptions:
            continue
        vid_filename = os.path.join(config.avi_video_root_path, vid_name + '.avi') 
        vid_idt_filename = os.path.join(config.idt_raw_root_path, vid_name)
        cmd="./DenseTrackStab  %s -W 15 -s 6 -t 6 | gzip > %s"%(vid_filename,vid_idt_filename)
        while len(threading.enumerate()) >= max_th:
            pass 
   
        print("Append another fine to the thread pool %s from %s to %s"%(vid_name, vid_filename, vid_idt_filename), (time.time() - st) * 1.0 / 60  )
        now_th = threading.Thread(target=cmd_runner, args = [cmd])
        now_th.start()
        thread_pool.append(now_th)
    
    for th in thread_pool:
        th.join()
    print "Finishing extracting improved DenseTrajecotory features..."


#   the core function for generating bag-of-words representation
def idt_bow_runner(vid_name,vid_raw_idt_filename,vid_idt_filename,num_centers,kmeans):
    pass
    bow_feat = np.zeros((1, num_centers))
    
    # parse the raw improved dense trajectory feature file...
    with gzip.GzipFile(vid_raw_idt_filename, "r") as f:
        pass
        line_count = 0

        # each line is a trajectory feature vector
        for now_raw_line in f:
            now_raw_line = now_raw_line.replace("\r", "")
            now_raw_line = now_raw_line.replace("\n", "")
            now_raw_line = now_raw_line.replace(" ", "")
            tmp_str = now_raw_line.split("\t")

            # the first 10 vector values are position information about current trajectory, which should be
            # discarded...
            tmp_str = tmp_str[10:-1]
            feat_vec = np.array(tmp_str)
            feat_vec = np.transpose(feat_vec)
            feat_vec = np.expand_dims(feat_vec, axis=0)

            # predict the k-means center, which current trajectory feature belongs to...
            pred_ind = kmeans.predict(feat_vec).tolist()[0]
            bow_feat[0, pred_ind] += 1
            # print "bow in line:", line_count, " vid: ", vid_name
            line_count += 1

            # for some super long videos, we do not need the full trajectories information, just break after line_count>=threshold
            if line_count>=5000000:
                break
            pass

        # Bag-of-Words feature normalization...
        bow_feat = bow_feat / (np.sum(bow_feat)+0.01)

    # save bag-of-words IDT feature vector to file...
    np.save(vid_idt_filename, bow_feat)
    return


def gen_idt_bow_feat(config,rev_mode=False, exceptions = None):

    # load the k-means clustering centers from pickle file...
    kmeans = utils.read_object_from_pkl(config.idt_codebook_filename)
    num_centers = len(kmeans.cluster_centers_)

    # path for bow encoded features and raw IDT featureS
    idt_feat_root_path=config.idt_bow_root_path
    idt_raw_feat_root_path = config.idt_raw_root_path

    if not os.path.exists(idt_feat_root_path):
        os.mkdir(idt_feat_root_path)

    
    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)


    # generate bag-of-words representations for videos IDT features using multiple threads handling...
    thread_pool=[]
    for now_video_label in all_video_label_list:
        vid_name=now_video_label[0]
        if vid_name in exceptions:
            continue
        #   the original improved dense trajectory file...
        vid_raw_idt_filename=os.path.join(idt_raw_feat_root_path,vid_name+config.idt_raw_file_format)

        #   the bag-of-words representation file...
        vid_idt_filename = os.path.join(idt_feat_root_path, vid_name + config.idt_bow_file_format)

        print "from: ",vid_raw_idt_filename,"---> to: ",vid_idt_filename

        #   if the bag-of-words representation file already existed, skip...
        if os.path.isfile(vid_idt_filename):
          continue

        # block starting new threads, if current thread_pool is full
        while len(threading.enumerate())>=max_th:
            pass

        # initiate a new thread for bag-of-words representation generation... 
        now_th=threading.Thread(target=idt_bow_runner,args=[vid_name,vid_raw_idt_filename,vid_idt_filename,num_centers,kmeans])
        now_th.start()
        thread_pool.append(now_th)


    # wait all threads to be finished...
    for th in thread_pool:
        th.join()

if __name__== "__main__":
    pass

    # import your own homework 2 configuration files here.

    import configs.hw2_config as config
    root_path = os.path.join(config.dataset_root_path, 'med_mini_idt_bow_partial')
    fnames = os.listdir(root_path)
    exceptions = set([fname[:-4] for fname in fnames])
    # feat_ext(config, exceptions)
    gen_idt_bow_feat(config,rev_mode=False, exceptions=exceptions)
