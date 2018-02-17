'''
    Converting mp4 format MED videos to avi format, so that dense trajectory feature extraction program could work on it.
'''

import sys
sys.path.append("../")
import os
import utils

def convert_all(config, exceptions):
    vid_root_path=config.video_root_path
    avi_root_path=config.avi_video_root_path

    if not os.path.exists(avi_root_path):
        os.mkdir(avi_root_path)

    # get the combined training and testing video list
    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)

    print len(all_video_label_list)
    cnt = 0
    for now_video_label in all_video_label_list:
        vid_name=now_video_label[0]
        if vid_name in exceptions:
            continue
        # vid_full_filename=os.path.join(vid_root_path,vid_name+config.video_file_format)
        vid_full_filename=os.path.join('/home/ubuntu/11775_data/clips',vid_name+config.video_file_format)
        print "Converting: ",vid_full_filename,os.path.isfile(vid_full_filename)
        dest_full_filename=os.path.join(avi_root_path,vid_name+config.avi_file_format)
        if os.path.isfile(dest_full_filename):
            print "Existed: ",dest_full_filename
            continue
        cnt += 1
        print("Converting the %dth video to %s"%(cnt, dest_full_filename))
        cvt_command="ffmpeg -i %s -vcodec copy -acodec copy %s"%(vid_full_filename,dest_full_filename)
        os.system(cvt_command)

if __name__== "__main__":
    pass
    import configs.hw2_config as config
    root_path = os.path.join(config.dataset_root_path, 'med_mini_idt_bow_partial')
    fnames = os.listdir(root_path)
    exceptions = set([fname[:-4] for fname in fnames])
    convert_all(config, exceptions)
    print "done."
