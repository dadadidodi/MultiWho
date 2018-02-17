#   the basic dataset configurations of our provided pipeline for homework 1
#   you could change anything according your own pipeline design (just leave a readme file in ~/11775_code/hw1/)

import os

#   the root path of the dataset
dataset_root_path="/home/ubuntu/11775_data/"
assert os.path.exists(dataset_root_path)

#   the root path which stores train/validation/test video list
trn_tst_list_root_path=os.path.join(dataset_root_path,"med_mini_list")
assert os.path.exists(trn_tst_list_root_path)

#   train/validation/test video list files (each line is a tuple of (video_name,label))
all_train_list_filename=os.path.join(trn_tst_list_root_path,"all_trn.lst")
all_val_list_filename=os.path.join(trn_tst_list_root_path,"all_val.lst")
all_test_list_filename=os.path.join(trn_tst_list_root_path,"all_tst_fake.lst")

assert os.path.isfile(all_train_list_filename);assert os.path.isfile(all_val_list_filename);assert os.path.isfile(all_test_list_filename)

#   the root path of all videos in mp4 format
video_root_path=os.path.join(dataset_root_path,"med_mini_video")
video_file_format=".mp4"
assert os.path.exists(video_root_path)

ds_video_root_path = os.path.join(dataset_root_path, "down_samp_video")
avi_video_root_path = os.path.join(dataset_root_path, "idt_clip_raw")
avi_file_format = ".avi"
idt_raw_root_path = os.path.join(dataset_root_path, "idt_clip_raw_npy")
idt_raw_file_format = ''
idt_bow_root_path = os.path.join(dataset_root_path, 'idt_bow_30')
idt_bow_file_format = '.npy'
idt_bow_full_path = os.path.join(dataset_root_path, "med_mini_idt_bow_partial")
#   the root path of the audio track (wav format) for all videos (you may change it according your own pipeline design)
audio_root_path=os.path.join(dataset_root_path,"med_mini_audio")
audio_file_format=".wav"

#   the root path of our provided CMU Sphinx ASR transcriptions (in pkl format)
cmu_asr_root_path=os.path.join(dataset_root_path,"med_mini_text")
cmu_asr_file_format=".pkl"
assert os.path.exists(cmu_asr_root_path)

#   the root path of bag-of-words representation for ASR features
asr_bow_root_path=os.path.join(dataset_root_path,"med_mini_asr_bow")
asr_bow_file_format=".npy"

#   corresponding text vocabulary book
cmu_asr_vocabbook_filename=os.path.join(dataset_root_path,"vocab.pkl")
assert os.path.isfile(cmu_asr_vocabbook_filename)
idt_codebook_filename = os.path.join(dataset_root_path, "med_mini_idt_codebook.pkl")
#   event id and corresponding name
event_id_name_dict={'P001':'assembling_shelter',
                    'P002':'batting_in_run',
                    'P003':'making_cake'}
