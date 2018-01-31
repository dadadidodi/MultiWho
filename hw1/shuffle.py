import mfcc_clustering
import configs.hw1_config as config
import utils
from random import shuffle
import numpy as np

import os
nb = input("RANDOM SEED FOR SHUFFLING: ")
try:
    number = int(nb)
except ValueError:
    print("Invalid Number")

full_lists = utils.get_video_and_label_list(config.all_train_list_filename) + \
        utils.get_video_and_label_list(config.all_val_list_filename)
np.random.shuffle(full_lists)
new_path = config.trn_tst_list_root_path
print("...........WRITE TO %s"%new_path)
print(len(full_lists))
if not os.path.exists(new_path):
    os.mkdir(new_path)


train = []
val = []
p = [0] * 4
for x, y in full_lists:
    if y[-1] == 'L':
        if p[0] < 100:
            train.append((x, y))
            p[0] += 1
        else:
            val.append((x, y))

    elif y[-1] == '1':
        if p[1] < 10:
            train.append((x, y))
            p[1] += 1
        else:
            val.append((x, y))

    elif y[-1] == '2':
        if p[2] < 10:
            train.append((x, y))
            p[2] += 1
        else:
            val.append((x, y))

    elif y[-1] == '3':
        if p[3] < 10:
            train.append((x, y))
            p[3] += 1
        else:
            val.append((x, y))
    else:
        print("WRONG", y)
print(p)
print("Shuffle into training with %d samples and validating with %d samples"%(len(train), len(val)) )
with open(os.path.join(new_path, 'all_trn.lst'), 'w') as f:
    f.write('\n'.join([ x + ' ' + y for x, y in train]))
with open(os.path.join(new_path, 'all_val.lst'), 'w') as f:
    f.write('\n'.join([ x + ' ' + y for x, y in val]))





