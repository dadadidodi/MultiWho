"""
    A reference code for down-sampling MED original video files...
    The default setting selects the first 60 seconds from the original video files with 15fps down-sampling.
"""

import os
import sys
sys.path.append("../")
import utils
import threading

max_th=5

def cmd_runner(cmd):
    os.system(cmd)

def video_downsample(config,ds_vid_len,ds_vid_frame_rate):
    if not os.path.exists(config.ds_video_root_path):
        os.mkdir(config.ds_video_root_path)

    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)

    thread_pool=[]

    for now_video_label in all_video_label_list:

        vid_name=now_video_label[0]
        vid_filename=os.path.join(config.video_root_path,vid_name+config.video_file_format)
        ds_vid_filename=os.path.join(config.ds_video_root_path,vid_name+config.video_file_format)

        if os.path.isfile(ds_vid_filename):
            continue

        assert(os.path.isfile(vid_filename))
        print "Down-sampling video : ",vid_filename

        ffmpeg_cmd="ffmpeg -y -ss 0 -i %s -strict experimental -t %d -r %d %s"%(vid_filename,ds_vid_len,ds_vid_frame_rate,ds_vid_filename)
        print ffmpeg_cmd

        while len(threading.enumerate())>=max_th:
            pass

        now_th=threading.Thread(target=cmd_runner,args=[ffmpeg_cmd])
        now_th.start()
        thread_pool.append(now_th)

    for th in thread_pool:
        th.join()


if __name__== "__main__":
    pass
    import configs.hw2_config as config

    #down-sample frame length (seconds)
    ds_vid_len=10

    #down-sample video frame rate
    ds_vid_frame_rate=15
    
    video_downsample(config,ds_vid_len,ds_vid_frame_rate)
