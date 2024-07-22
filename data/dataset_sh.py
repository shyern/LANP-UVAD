import numpy as np
import os
import glob
from data.base_dataset import BaseDataset
from data.normprop import normality_propagation

class Dataset_SH(BaseDataset):
    def initialize(self, args, sample_type="uniform", is_train=True, is_normal=True, eval_train=False):
        self.dataset_name = args.dataset
        self.seg_len = args.segment_len
        self.normprop_scores_path = args.np_scores_path
        self.process_len = args.process_len
        self.sample_type = sample_type
        self.is_train = is_train
        self.is_normal = is_normal
        self.eval_train = eval_train
        self.r = args.r
        self.h = args.h
        self.anno_path = args.anno_path
        self.feature_path = args.feature_path
        self.feature_name_end = args.feature_name_end
        self.logger_info = None
        self.video_info_dict = {}
        self.test_anno_dict = {}

        if self.is_train or (self.is_train == False and self.eval_train == True):
            self.video_list = open(args.training_split, 'r').readlines()
        else:
            self.video_list = open(args.testing_split, 'r').readlines()
        
        if self.normprop_scores_path is not None:
            self.normprop_scores_dict = np.load(self.normprop_scores_path, allow_pickle=True).item()
        else:
            self.normprop_scores_dict = None

        self.parser_anno()
        self.parser_info()

    def parser_anno(self):
        # for Shanghai-Tech test and train video annotations.
        assert os.path.exists(self.anno_path), "The annotation path does not exist."
        anno_paths = glob.glob(os.path.join(self.anno_path, '*.npy'))
        for a_path in anno_paths:
            video_name = a_path.strip().split('/')[-1].split('.')[0]
            anno_ = np.load(a_path)
            if len(anno_) % self.seg_len == 0:
                annotation = anno_
            else:
                anno_last = anno_[-1]
                annotation = list(anno_)
                for i in range(len(anno_), (len(anno_)//self.seg_len+1)*self.seg_len):
                    annotation.append(anno_last)
            self.test_anno_dict[video_name] = np.array(annotation).astype(np.float32)
   
    def parser_info(self):
        if self.is_train:
            video_name_list = []
            score_v_list = []
            gts_list = []
            nor_idx_v_gt = []
            abnormal_num_v = int(len(self.video_list)*0.27)
            normal_num_v = len(self.video_list) - abnormal_num_v
            # np_scores_dict = {}
            for i, item in enumerate(self.video_list):
                video_name, _, frame_len = item.strip().split(',')
                feat_path = os.path.join(self.feature_path, video_name+self.feature_name_end)
                feature_ori = np.load(feat_path)
                T = feature_ori.shape[0]
                abn_num=int(self.r*T)
                if len(feature_ori.shape) == 3:
                    feature_ori = np.mean(feature_ori, axis=1)
                if self.normprop_scores_dict is not None:
                    Z, score_v = self.normprop_scores_dict[video_name]['pseudo_label_scores'], self.normprop_scores_dict[video_name]['score_v']
                    sorted_idxs_F = np.argsort(Z)
                    abn_idxs = sorted_idxs_F[:abn_num]
                    pseudo_label = np.zeros(T)
                    pseudo_label[abn_idxs] = 1
                else:
                    Z, pseudo_label, score_v = normality_propagation(feature_ori, abn_num=abn_num, is_ucf=False)
                    # np_scores_dict[video_name] = {'pseudo_label_scores': Z, 
                    #                               'score_v': score_v}

                video_name_list.append(video_name)
                score_v_list.append(score_v)
                is_normal = (len(video_name) == 6)
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
            # np.save('./pseudo_label_scores_sh.npy', np_scores_dict)
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
                        if video_name in self.test_anno_dict.keys():
                            test_anno = self.test_anno_dict[video_name]
                            label_test = np.array([test_anno[i:i+self.seg_len] for i in np.arange(0, len(test_anno), self.seg_len)])
                        else:
                            label_test = np.zeros((snipts_len, self.seg_len))
                        
                info = {'feature': np.load(feat_path),
                     'label_video': video_label,
                     'label_test': label_test}
                self.video_info_dict[video_name] = info