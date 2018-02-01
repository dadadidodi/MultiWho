import sys
sys.path.append("../")
import os
import utils
import librosa
import numpy as np
#   extract the audio track from original mp4 videos using ffmpeg
def extract_audio(config):
    pass
    if not os.path.exists(config.audio_root_path):
        os.mkdir(config.audio_root_path)

    #   read and concatenate train/validation/test video lists
    all_video_label_list=utils.get_video_and_label_list(config.all_train_list_filename)+\
                         utils.get_video_and_label_list(config.all_val_list_filename)+\
                         utils.get_video_and_label_list(config.all_test_list_filename)

    cnt = 0
    print('TOT NUMBER: %d' % (len(all_video_label_list)))
    for now_video_label in all_video_label_list:
        cnt += 1
        vid_name=now_video_label[0]
        vid_full_fn=os.path.join(config.video_root_path,vid_name+config.video_file_format)
        print("In processing: %d%%.....%s"% (int(100* cnt / len(all_video_label_list)), vid_full_fn) )

        videosize = os.stat(vid_full_fn).st_size
        if videosize <= 40000:
            audio_full_fn=os.path.join(config.audio_root_path,vid_name+config.audio_file_format)
            command = "ffmpeg -y -i %s -ac 1 -f wav %s -loglevel panic" % (vid_full_fn, audio_full_fn)
        else:
            audio_full_fn=os.path.join(config.audio_root_path,vid_name)
            command = "ffmpeg -y -i %s -ac 1 -segment_time 240 -f segment %s_%%03d.wav -loglevel panic" % (vid_full_fn, audio_full_fn)
        #   call command line for audio track extraction
        os.system(command)


#TODO: extract the mfcc feature from audio files
def extract_mfcc(config):
    pass
    if not os.path.exists(config.mfcc_root_path):
        os.mkdir(config.mfcc_root_path)

    #   read and concatenate train/validation/test video lists
    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename)
    cnt = 0
    ignore = 0
    #   you may add your code to extract mfcc features and store them in numpy files...
    for now_video_label in all_video_label_list:
        vid_name = now_video_label[0]
        audio_full_fn = os.path.join(config.audio_root_path, vid_name + config.audio_file_format)
        cnt += 1
        print("In processing: %d%.....%s"% (int(100* cnt / len(now_video_label)), audio_full_fn) )
        if not os.path.exists(audio_full_fn):
            print("........Ignore", audio_full_fn)
            continue
        y, sr = librosa.load(audio_full_fn)
        mfcc = librosa.feature.mfcc(y = y, sr = sr, hop_length = 512, n_mfcc = 20)
        mfcc_full_fn = os.path.join(config.dataset_root_path, vid_name)
        np.save(mfcc_full_fn, mfcc)

if __name__=="__main__":
    pass