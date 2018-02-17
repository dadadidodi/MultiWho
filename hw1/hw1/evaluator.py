import sys
sys.path.append("../")
import os
import utils as utils
from sklearn.metrics import average_precision_score
import configs.hw1_config as config
#   a demo code illustrating how to use sklearn to evaluate average precision scores given ground-truth labels
#   and prediction score files. You could use this function to evaluate your performance on the all_val.lst video list.
def evaluate_ap(config):

    #   load the ground-truth file list
    gt_list_fn="example_gt_and_pred/gt.lst"
    test_video_label_list=utils.get_video_and_label_list(gt_list_fn)
    val = 0
    for event_id, event_name in config.event_id_name_dict.iteritems():
        print "Evaluating the average precision (AP) with classifier ",event_id," name: ",event_name,"..."

        #   load the outputted prediction score files to calculate the average precision
        event_pred_score_fn = os.path.join("example_gt_and_pred", event_id+"_pred_score.lst")
        y_score = utils.read_score_list_from_file(event_pred_score_fn)

        y_gt=[]
        for now_video_label in test_video_label_list:
            vid_gt_label=now_video_label[1]
            if vid_gt_label==event_id:
                y_gt.append(1)
            else:
                y_gt.append(0)

        #   the number of ground-truths and the number of prediction scores should be same
        assert(len(y_gt)==len(y_score))
        val += average_precision_score(y_gt, y_score)
        print "Average precision: ",average_precision_score(y_gt,y_score)
        
    print "Finish evaluating the average precision (AP) metric on all classifiers...", val / 3.0


if __name__=="__main__":
    evaluate_ap(config)
