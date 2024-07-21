import numpy as np
import os
from tqdm import tqdm
from data.base_dataset import BaseDataset
from data.normprop import normality_propagation

class Dataset_UCF(BaseDataset):
    def initialize(self, args, sample_type="uniform", is_train=True, is_normal=True, eval_train=False):
        self.dataset_name = args.dataset
        self.seg_len = args.segment_len
        self.labelprop_scores_path = args.np_scores_path
        self.process_len = args.process_len
        self.sample_type = sample_type 
        self.is_train = is_train
        self.is_normal = is_normal
        self.eval_train = eval_train
        self.r = args.r
        self.h = args.h
        self.feature_path = args.feature_path
        self.feature_name_end = args.feature_name_end
        self.logger_info = None
        self.video_info_dict = {}
        self.test_anno_dict = {}

        if self.is_train or (self.is_train == False and self.eval_train == True):
            self.video_list = open(args.training_split, 'r').readlines()
        else:
            self.video_list = open(args.testing_split, 'r').readlines()
            self.anno_path = args.anno_path
            self.parser_test_anno()

        if self.labelprop_scores_path is not None:
            self.labelprop_scores_dict = np.load(self.labelprop_scores_path, allow_pickle=True).item()
        else:
            self.labelprop_scores_dict = None

        self.parser_info()
    
    def parser_test_anno(self):
        # for UCF-Crimes test video annotations.
        test_dict = {}
        lines = open(self.anno_path, 'r').readlines()
        for line in lines:
            line_list = line.strip().split('\t')
            annotations = []
            for i in range(2, 6):
                if int(line_list[i]) == -1:
                    break
                else:
                    annotations.append(int(line_list[i]))

            test_dict[line_list[0]] = [line_list[1], annotations, int(line_list[-1])]

        # for UCF-Crimes test video annotations.
        for key in test_dict.keys():
            ano_type, anno, frames_num = test_dict[key]
            anno_ = np.zeros(frames_num).astype(np.float32)
            if len(anno) >= 2:
                front = anno[0]
                back = anno[1]
                if front < anno_.shape[0]:
                    anno_[front:min(back, anno_.shape[0])] = 1
            if len(anno) == 4:
                front = anno[-2]
                back = anno[-1]
                if front < anno_.shape[0]:
                    anno_[front:min(back, anno_.shape[0])] = 1

            if len(anno_) % self.seg_len == 0:
                annotation = anno_
            else:
                anno_last = anno_[-1]
                annotation = list(anno_)
                for i in range(len(anno_), (len(anno_)//self.seg_len+1)*self.seg_len):
                    annotation.append(anno_last)

            self.test_anno_dict[key[:-4]] = annotation
     
    def parser_info(self):        
        if self.is_train:
            video_name_list = []
            score_v_list = []
            gts_list = []
            nor_idx_v_gt = []
            abnormal_num_v = int(len(self.video_list)*0.5)
            normal_num_v = len(self.video_list) - abnormal_num_v
            # np_scores_dict = {}
            for i, item in enumerate(tqdm(self.video_list)):
                video_name, _, frame_len = item.strip().split(',')
    
                feat_path = os.path.join(self.feature_path, video_name+self.feature_name_end)
                feature_ori = np.load(feat_path)  # T, nc, dim
                T = feature_ori.shape[0]
                abn_num=int(self.r*T)
                if self.labelprop_scores_dict is not None:
                    Z, score_v = self.labelprop_scores_dict[video_name]['pseudo_label_scores'], self.labelprop_scores_dict[video_name]['score_v']
                    sorted_idxs_F = np.argsort(Z)
                    abn_idxs = sorted_idxs_F[:abn_num]
                    pseudo_label = np.zeros(T)
                    pseudo_label[abn_idxs] = 1
                else:
                    Z, pseudo_label, score_v = normality_propagation(feature_ori, abn_num=abn_num, is_ucf=True)
                    # np_scores_dict[video_name] = {'pseudo_label_scores': Z, 
                    #                               'score_v': score_v}

                video_name_list.append(video_name)
                score_v_list.append(score_v)

                is_normal = ('Normal' in video_name)
                gt_v = 0 if is_normal else 1
                gts_list.append(gt_v)
                if is_normal: nor_idx_v_gt.append(i)

                sample_idxs = self.uniform_sampling(feature_ori.shape[0])
                pseudo_label = pseudo_label[sample_idxs]
                feature_ori = feature_ori[sample_idxs]
                reweight = np.ones_like(pseudo_label)

                info = {'feature': feature_ori,
                    'pseudo_label': pseudo_label,
                    'reweight': reweight,
                    'high_confidence_norvideo': 0}
                self.video_info_dict[video_name] = info
            
            video_name_list = np.array(video_name_list)
            score_v_list = np.array(score_v_list)
            gts_list = np.array(gts_list)
            sorted_idxs = np.argsort(score_v_list)  # from small to large
            abn_idx_v = sorted_idxs[-abnormal_num_v:]
            nor_idx_v = sorted_idxs[:normal_num_v]
            
            nor_idx_v_high_cofidence = sorted_idxs[:int(normal_num_v*self.h)]
            nor_video_names = video_name_list[nor_idx_v]
            nor_video_names_high_confidence = video_name_list[nor_idx_v_high_cofidence]

            for video_name in nor_video_names:
                self.video_info_dict[video_name]['pseudo_label'] = np.zeros(len(self.video_info_dict[video_name]['pseudo_label']))

            for video_name in nor_video_names_high_confidence:
                self.video_info_dict[video_name]['high_confidence_norvideo'] = 1

            pred_video = np.array([1]*len(abn_idx_v) + [0]*len(nor_idx_v))
            gt_video = np.concatenate((gts_list[abn_idx_v], gts_list[nor_idx_v]))

            tp = np.sum((pred_video)*(gt_video))
            tn = np.sum((1-pred_video)*(1-gt_video))
            fp = np.sum(pred_video*(1-gt_video))
            fn = np.sum((1-pred_video)*(gt_video))

            self.logger_info = "tp: {}/{:.2f}, tn: {}/{:.2f}, fp: {}, fn: {}".format(tp, tp/len(abn_idx_v), tn, tn/len(nor_idx_v), fp, fn)
            # np.save('./pseudo_label_scores_ucf.npy', np_scores_dict)
        else:
            for item in self.video_list:
                video_name, video_label, frame_len = item.strip().split(',')
                feat_path = os.path.join(self.feature_path, video_name+self.feature_name_end)
                frame_len = int(frame_len)
                if frame_len % self.seg_len == 0:
                    snipts_len = frame_len // self.seg_len
                else:
                    snipts_len = frame_len // self.seg_len + 1

                if self.eval_train:
                    label_test = np.zeros(snipts_len)
                else:
                    if video_label == 0:
                        label_test = np.zeros((snipts_len, self.seg_len))
                    else:
                        test_anno = self.test_anno_dict[video_name]
                        label_test = np.array([test_anno[i:i+self.seg_len] for i in np.arange(0, len(test_anno), self.seg_len)])
                info = {'feature': np.load(feat_path),
                     'label_video': video_label,
                     'label_test': label_test}
                self.video_info_dict[video_name] = info
