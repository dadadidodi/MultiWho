# some common utility functions you could use and extend for your homework

import pickle

def get_video_and_label_list(list_fn):
    ret_list=[]
    with open(list_fn,"r") as f:
        all_lines=f.readlines()
    for now_line in all_lines:
        tmp_str=now_line.split()
        vid_name=tmp_str[0];vid_label=tmp_str[1]
        ret_list.append((vid_name,vid_label))

    return ret_list

def write_score_list_to_file(to_write_list,out_fn):
    to_write_list=[str(i)+"\n" for i in to_write_list]
    with open(out_fn,"w") as f:
        f.writelines(to_write_list)

def read_score_list_from_file(fn):
    ret_score_list=[]
    with open(fn,"r") as f:
        all_lines=f.readlines()
    for now_line in all_lines:
        now_line.replace("\n","")
        now_line.replace("\r","")
        ret_score_list.append(float(now_line))
    return ret_score_list

def write_object_to_pkl(object,fn):
    with open(fn,"wb") as f:
        pickle.dump(object,f)

def read_object_from_pkl(fn):
    with open(fn,"rb") as f:
        obj=pickle.load(f)
    return obj