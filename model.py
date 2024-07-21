import torch.nn as nn
import torch
import numpy as np

class AD_Model(nn.Module):
    def __init__(self, len_feature, feature_embed, dropout_rate=0.6):
        super(AD_Model, self).__init__()
        self.len_feature = len_feature
        self.feature_embed = feature_embed
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=self.feature_embed, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Linear(self.feature_embed, 512), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(512, 1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        bs, ncrops, t, f = x.shape
        embeddings = x.view(-1, t, f)
        embeddings = embeddings.permute(0, 2, 1) # [b, t, d] --> [b, d, t]
        embeddings = self.f_embed(embeddings) # temporal convolution
        embeddings = embeddings.permute(0, 2, 1) # [b, d, t] --> [b, t, d]
        out = self.dropout(embeddings)
        out = self.f_cls(out)
        logits = out.view(bs, ncrops, -1).mean(1)

        return logits

# cm
class Memory_module(nn.Module):
    def __init__(self, extracted_features=None, video_name_all=None, video_names_nor_hc=None, pseudo_labels=None):
        super(Memory_module, self).__init__()
        self.extracted_features = extracted_features  # N*32*2048 (video_num, ncrop, process_len, dim)
        self.video_name_all = video_name_all
        self.video_names_nor_hc = video_names_nor_hc
        self.pseudo_labels = pseudo_labels  # N*32 (video_num, process_len)

        self.video_num, ncrop, self.process_len, self.dim = self.extracted_features.shape
        self.logger_info = None

        self.normal_memory = self.build_memory(self.video_names_nor_hc)

    def build_memory(self, video_names_hc):
        memory = []
        for video_name in video_names_hc:
            idx_v = np.where(self.video_name_all == video_name)[0][0]
            features = self.extracted_features[idx_v].squeeze()  # all clips in the normal video
            features = torch.mean(features, dim=-2).squeeze()
            memory.append(features)

        memory = torch.stack(memory, 0) 

        return memory

    def update_dataloader(self, train_dataset_loader):
        device = self.extracted_features.device
        for i in range(self.video_num):
            video_name = self.video_name_all[i]
            feature_i = self.extracted_features[i]

            if train_dataset_loader.dataset.dataset_name == 'ucf-crime': # use cosine distance (1-cosine similarity) for UCF-Crime
                dist = nn.CosineSimilarity(dim=-1, eps=1e-6)(feature_i.repeat(self.normal_memory.shape[0],1,1), self.normal_memory.unsqueeze(1).repeat(1,32,1))
                dist = 1 - dist
            else:  # use Euclidean distance for ShanghaiTech
                dist = torch.pow((torch.sum((feature_i-self.normal_memory.unsqueeze(1))**2, axis=-1) / (self.dim)), 1/2)
            dist, indices = torch.min(dist, dim=0)
            
            reweight = torch.exp(-torch.abs(dist-self.pseudo_labels[i].to(device))) 
            train_dataset_loader.dataset.video_info_dict[video_name]['reweight'] = reweight.cpu().numpy()
        
        updated_tag = True
        return updated_tag