#   A "random" bag-of-words implementation using the ASR feature, you should modify this code and implement the
#   real bag-of-words algorithm.

import os
import sys
sys.path.append("../")
import utils
import pickle
import numpy as np

def get_bow_vec(config):

    #   read and concatenate train/validation/test video lists
    all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename)

    #   the number of bag-of-words centers
    if not os.path.exists(config.asr_bow_root_path):
        os.mkdir(config.asr_bow_root_path)

    vocab_book=utils.read_object_from_pkl(config.cmu_asr_vocabbook_filename)

    word_len=len(vocab_book)
    for now_video_label in all_video_label_list:
        vid_name = now_video_label[0]
        asr_filename=os.path.join(config.cmu_asr_root_path,vid_name+config.cmu_asr_file_format)
        asr_bow_filename=os.path.join(config.asr_bow_root_path,vid_name+config.asr_bow_file_format)

        word_list=utils.read_object_from_pkl(asr_filename)

        asr_bow_vec=np.zeros((1,word_len))

        #   we randomly set the Bag-of-Words representation vector
        #   according to the number of words in ASR transcription file (this is absolutely ridiculous:)
        asr_bow_vec[0,len(word_list)%word_len]=1

        np.save(asr_bow_filename,asr_bow_vec)


if __name__=="__main__":
    pass
