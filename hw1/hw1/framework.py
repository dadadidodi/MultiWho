# The exemplary pipeline code for 11-775 Homework 1

import sys
sys.path.append("../")
import configs.hw1_config as config
import time
import audio_mfcc_feature_extraction as audio_mfcc
import evaluator as eval
import asr_bag_of_words as asr_bow

if __name__=="__main__":
    print "Start MED exemplary pipeline..."
    start_time=time.clock()

    #TODO:  audio track and MFCC feature extraction

    audio_mfcc.extract_audio(config)
    audio_mfcc.extract_mfcc(config)

    #   bag-of-words feature representation based on ASR features
    asr_bow.get_bow_vec(config)

    #TODO: bag-of-words feature representation based on MFCC features

    #TODO: svm training and testing,

    #  an example on evaluating average precision
    eval.evaluate_ap(config)

    elapsed_time=(time.clock()-start_time)
    print "Finish MED exemplary pipeline, running time: ",elapsed_time,"s..."