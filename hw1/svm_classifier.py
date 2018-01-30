import sys
sys.path.append("../")
import os
import utils as utils
import numpy as np
from sklearn.svm import SVC

# TODO: implement the SVM classifier training function here.
# it should load different encoded videos features (MFCC, ASR, etc.) in all_trn.lst (validation)
# or all_trn.lst+all_val.lst (final submission) and train corresponding classifiers for each event types.

def train(args):
    pass

# TODO: implement the SVM classifier testing (predicting) function here
# it should load classifiers of each event-feature combination and output prediction score
# for each video to a score file. For example, to submit results of videos in all_tst_fake.lst with MFCC feature
# using P001 event classifier. The test function should finally output the score file as P001_mfcc.lst. This is
# exactly what you should include under the "scores/" path in your ANDREWID_HW1.zip submission.

def test(args):
    pass


if __name__=="__main__":
    pass
