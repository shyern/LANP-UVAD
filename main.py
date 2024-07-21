import torch

import os
import numpy as np
import yaml
import argparse

from utils import *
from model import AD_Model, Memory_module
from loss import Loss_bce
from data.dataset_loader import CreateDataset

def train(dataloader, model, optimizer_model, criterion, epoch, device):
    with torch.set_grad_enabled(True):
        model.train()

        for features, pseudo_labels, reweight, _, _ in dataloader:
            bs, nc, t, dim = features.shape
            features = features.type(torch.float).to(device)
            pseudo_labels = pseudo_labels.type(torch.float).to(device)
            reweight = reweight.type(torch.float).to(device)
            scores = model(features)
            loss_cls = criterion(scores, pseudo_labels, reweight)

            optimizer_model.zero_grad()
            loss_cls.backward()
            optimizer_model.step()

        logger.info('Epoch: [{:.0f}/{:.0f}], '
                    'sample_num: {}, '
                    'loss_cls: {:.2f}.'.format(epoch, args.epochs, bs*t*nc, loss_cls))

def test(model, test_loader, device, is_train_sample=False):
    if is_train_sample:
        with torch.no_grad():
            model.eval()
            losses_dict={}
            for features, label_frames, video_name in test_loader:
                features = features.type(torch.float).to(device)
                label_frames = label_frames.type(torch.float).to(device)
                outputs = model(features)
                scores = outputs.squeeze().cpu().numpy()
        
                losses_dict[video_name[0]] = scores

        logger.info('Eval abnormal videos in training set. Finished!')

        return losses_dict
    else:
        with torch.no_grad():
            model.eval()
            total_scores = []
            total_labels = []
            scores_dist = {}
            for features, label_frames, video_name in test_loader:
                features = features.type(torch.float).to(device)
                label_frames = label_frames.type(torch.float).to(device)
                outputs = model(features)

                scores = outputs.squeeze().cpu().numpy()
                scores_dist[video_name[0]] = scores

                for score, label in zip(scores, label_frames[0]):
                    score = [score] * args.segment_len
                    label = label.detach().cpu().numpy().astype(int).tolist()
                    total_scores.extend(score)
                    total_labels.extend(label)

        total_score_frames = np.array(total_scores)
        total_label_frames = np.array(total_labels)

        prauc_frames, rocauc_frames = calc_metrics(total_score_frames, total_label_frames)
    
        logger.info('Testing: pr@ {:.2f}%, '
              'auc@ {:.2f}% \t'.format(prauc_frames, rocauc_frames))
        
        return scores_dist, prauc_frames, rocauc_frames

def prepare_log_files(args):
    param_str = '{}_lr_{}_{}'.format(args.dataset, args.lr, get_timestamp())
    
    ckpt_path = os.path.join(args.ckpt_path, args.dataset, param_str)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    logger_path = os.path.join(args.logger_path, args.dataset)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    logger = get_logger(logger_path+'/{}.txt'.format(param_str))
    logger.info('Train this model at time {}'.format(get_timestamp()))
    log_param(logger, args)
    logger.info(param_str)

    return logger, ckpt_path

def prepare_model(args, device):
    cls_model = AD_Model(args.feature_dim, 512, args.dropout_rate)

    if torch.cuda.is_available():
        cls_model.to(device)
        torch.backends.cudnn.benchmark = True

    return cls_model

def extract_features(dataloader):
    features_all = []
    video_name_all = []
    pseudo_labels_all = []
    normal_video_names_high_confidence = []
    for features, pseudo_labels, _, high_confidence_norvideo_tag, video_name in dataloader:
        if features.shape[1] != 1:
            features = torch.unsqueeze(torch.mean(features, dim=1), 1)
        features_all.append(features)
        video_name_all.append(video_name)
        pseudo_labels_all.append(pseudo_labels)
        if high_confidence_norvideo_tag: normal_video_names_high_confidence.append(video_name)
    
    features_all = torch.cat(features_all, 0)
    video_name_all = np.concatenate(video_name_all)
    pseudo_labels_all = torch.cat(pseudo_labels_all)
    normal_video_names_high_confidence = np.concatenate(normal_video_names_high_confidence)

    return features_all.type(torch.float32), video_name_all, pseudo_labels_all, normal_video_names_high_confidence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', 
                        dest='config_file',
                        help='The yaml configuration file')
    args, unprocessed_args = parser.parse_known_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.load(f, Loader=yaml.FullLoader))
    
    args = parser.parse_args(unprocessed_args)
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.device=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    set_seeds(args.seed)

    logger, ckpt_path = prepare_log_files(args)

    '''build model'''
    model = prepare_model(args, device)
    loss_criterion = Loss_bce()

    test_loader, train_loader, train_eval_loader, train_loader_cluster = CreateDataset(args, logger)
    
    '''load pretrained model'''
    if args.pretrained_path is not None:
        logger.info('load the pretrained model....')
        model.load_state_dict(torch.load(args.pretrained_ckpt))
        param_str_test = args.pretrained_ckpt.strip().split('/')[-2]
        scores_dict, _, _ = test(model, test_loader=test_loader, device=device)
        np.save('./test_results/{}/{}_test.npy'.format(args.dataset, param_str_test[-24:-5]), scores_dict)
        scores_dict = test(model=model, test_loader=train_eval_loader, device=device, is_train_sample=True)
        np.save('./test_results/{}/{}_train.npy'.format(args.dataset, param_str_test[-24:-5]), scores_dict)
        logger.info('load the pretrained model....finished!')

    optimizer_model = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.6)
     
    best_AUC, best_PR = 0, 0
    best_epoch_AUC, best_epoch_PR = 0, 0
    
    with torch.no_grad():
        features_all, video_name_all, pseudo_labels_all, video_names_nor_hc = extract_features(train_loader_cluster)

    memory = Memory_module(extracted_features=features_all.clone().to(device),
                            video_name_all=video_name_all, video_names_nor_hc=video_names_nor_hc,
                            pseudo_labels=pseudo_labels_all)
    updated_tag = memory.update_dataloader(train_loader)
    logger.info(memory.logger_info)
    
    for epoch in range(1, args.epochs + 1):

        train(train_loader, model, optimizer_model, loss_criterion, epoch, device)
            
        if epoch % args.test_freq == 0:
            scores_dist, test_prauc, test_rocauc = test(model=model, test_loader=test_loader, device=device)
            
            if test_rocauc > best_AUC:
                best_AUC = test_rocauc
                best_epoch_AUC = epoch
            if test_rocauc > args.th_auc*100:
                torch.save(model.state_dict(),
                            os.path.join(ckpt_path, 'epoch_{}_test_auc_{:.2f}_pr_{:.2f}.pkl'.
                                        format(epoch, test_rocauc, test_prauc)))
            if test_prauc > best_PR:
                best_PR = test_prauc
                best_epoch_PR = epoch
            if test_prauc > args.th_pr*100:
                torch.save(model.state_dict(),
                            os.path.join(ckpt_path, 'epoch_{}_test_auc_{:.2f}_pr_{:.2f}.pkl'.format(epoch, test_rocauc, test_prauc)))

            logger.info('best_AUC {:.2f} at epoch {}.\t best_PR {:.2f} at epoch {}.'.format(best_AUC, best_epoch_AUC, best_PR, best_epoch_PR))
            logger.info('============================')
