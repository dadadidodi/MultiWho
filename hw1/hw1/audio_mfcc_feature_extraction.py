import sys
sys.path.append("../")
import os
import utils

#   extract the audio track from original mp4 videos using ffmpeg
def extract_audio(config):
    pass
    if not os.path.exists(config.audio_root_path):
        os.mkdir(config.audio_root_path)

    #   read and concatenate train/validation/test video lists
    all_video_label_list=utils.get_video_and_label_list(config.all_train_list_filename)+\
                         utils.get_video_and_label_list(config.all_val_list_filename)+\
                         utils.get_video_and_label_list(config.all_test_list_filename)

    #   iterate over the video list and call system command for ffmpeg audio extraction
    for now_video_label in all_video_label_list:
        vid_name=now_video_label[0]
        vid_full_fn=os.path.join(config.video_root_path,vid_name+config.video_file_format)
        audio_full_fn=os.path.join(config.audio_root_path,vid_name+config.audio_file_format)

        #   call command line for audio track extraction
        command = "ffmpeg -y -i %s -ac 1 -f wav %s" % (vid_full_fn, audio_full_fn)
        os.system(command)


#TODO: extract the mfcc feature from audio files
def extract_mfcc(config):
    pass

    #   read and concatenate train/validation/test video lists
    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename)

    #   you may add your code to extract mfcc features and store them in numpy files...


if __name__=="__main__":
    pass
