import torch.utils.data as data
import torch
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt, is_train=True, is_normal=True, eval_train=False, transform=None, seed=1):
        pass
   
    def uniform_sampling(self, length):
        samples = np.arange(self.process_len) * length / self.process_len
        samples = np.floor(samples)
        return samples.astype(int)
    
    def __getitem__(self, index):
        video_name = list(self.video_info_dict.keys())[index]
        info = self.video_info_dict[video_name]

        features = info['feature']
        snipts_len = features.shape[0]

        if self.is_train:
            pseudo_label, high_confidence_norvideo = info['pseudo_label'], info['high_confidence_norvideo']
            reweight = info['reweight']

            if len(features.shape) == 2:
                features = np.expand_dims(features, 1)  # T, nc, dim
            else:
                features = features
            features = features.transpose(1, 0, 2)  # nc, T, dim
            
            return features, pseudo_label, reweight, high_confidence_norvideo, video_name
        else:
            label_frames = info['label_test']
            if len(label_frames) != snipts_len:
                label_frames = label_frames[:snipts_len]
            
            if len(features.shape) == 2:
                features = np.expand_dims(features, 1)
                
            features = features.transpose(1, 0, 2)  # nc, T, dim
            features = torch.from_numpy(features)
            label_frames = torch.from_numpy(label_frames).to(torch.float32)
            return features, label_frames, video_name

    def __len__(self):
        return len(self.video_info_dict.keys())

