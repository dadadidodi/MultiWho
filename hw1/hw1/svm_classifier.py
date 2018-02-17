import sys
sys.path.append("../")
import os
import utils as utils
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.linear_model import LogisticRegression
# TODO: implement the SVM classifier training function here.
# it should load different encoded videos features (MFCC, ASR, etc.) in all_trn.lst (validation)
# or all_trn.lst+all_val.lst (final submission) and train corresponding classifiers for each event types.
SEED = 23
def train(x, y, scale = True, C=100):
    scl = None
    if scale:
        print("Scaling before training svm")
        # scl = MinMaxScaler()
        scl = StandardScaler()
        scl.fit(x)
        x = scl.transform(x)
    clf = SVC(kernel ='rbf', C = C, probability = True, random_state = SEED)
    print(clf)
    clf.fit(x, y)
    return clf, scl

# TODO: implement the SVM classifier testing (predicting) function here
# it should load classifiers of each event-feature combination and output prediction score
# for each video to a score file. For example, to submit results of videos in all_tst_fake.lst with MFCC feature
# using P001 event classifier. The test function should finally output the score file as P001_mfcc.lst. This is
# exactly what you should include under the "scores/" path in your ANDREWID_HW1.zip submission.

def test(clf, x, scl = None):
    if scl is not None:
        x = scl.transform(x)
        print("scaling before testing")
    y = clf.predict(x)
    y_proba = clf.predict_proba(x)
    return y, y_proba


if __name__=="__main__":
    pass
