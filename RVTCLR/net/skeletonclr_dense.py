import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import numpy as np
from torch.nn.functional import normalize
import math
from torch import Tensor
import random
from itertools import permutations

class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, 
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          no_pretrain=True,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            self.tem_decay = 0.99999

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)


            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            
            self.register_buffer("queue2", torch.randn(feature_dim, queue_size))
            self.queue2 = F.normalize(self.queue2, dim=0)
            self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))
            
            self.count = torch.zeros(self.K, dtype=torch.int16).cuda()
            self.count2 = torch.zeros(self.K, dtype=torch.int16).cuda()

            self.value_transform = nn.Conv1d(128, 128, kernel_size=1, bias=False)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        self.count += 1
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
        self.count[ptr:ptr + batch_size] = 1
        
    def _dequeue_and_enqueue2(self, keys):
        self.count2 += 1
        batch_size = keys.shape[0]
        ptr = int(self.queue2_ptr)
        gpu_index = keys.device.index
        self.queue2[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
        self.count2[ptr:ptr + batch_size] = 1
        
        
    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    @torch.no_grad()
    def relative_tempo(self, im_q: Tensor, im_k: Tensor):
        B, C, T, V, M = im_q.shape
        random_indices = torch.randperm(B, device=im_q.device)
        selected_t1 = random_indices[:int(B * 0.5)] # in a batch！
        selected_t2 = random_indices[int(B * 0.5):]

        diff_tempo = random.choice([2])
        T_real = T // diff_tempo
        tempo1 = torch.arange(0, T, 1, device=im_q.device)[: T_real]
        tempo2 = torch.arange(0, T, diff_tempo, device=im_q.device)[ : T_real]
        im_q_real = torch.empty(B, C, T_real, V, M, device=im_q.device)
        im_k_real = torch.empty_like(im_q_real)
        im_k_negative = torch.empty_like(im_q_real)

        im_q_real[selected_t1] = im_q.index_select(0, selected_t1).index_select(2, tempo1)
        im_q_real[selected_t2] = im_q.index_select(0, selected_t2).index_select(2, tempo2)

        im_k_real[selected_t1] = im_k.index_select(0, selected_t1).index_select(2, tempo1)
        im_k_real[selected_t2] = im_k.index_select(0, selected_t2).index_select(2, tempo2)

        im_k_negative[selected_t1] = im_k.index_select(0, selected_t1).index_select(2, tempo2)
        im_k_negative[selected_t2] = im_k.index_select(0, selected_t2).index_select(2, tempo1)

        k_negative_A, k_negative_M, k_negative_dense, k_negative_dense_global, k_neg_backbone_feat = self.encoder_k(im_k_negative)
        k_negative_A = F.normalize(k_negative_A, dim=1)
        k_negative_M = F.normalize(k_negative_M, dim=1)
        k_negative_dense = F.normalize(k_negative_dense, dim=1)
        k_negative_dense_global = F.normalize(k_negative_dense_global, dim=1)

        return im_q_real, im_k_real, k_negative_A, k_negative_M, k_negative_dense, k_negative_dense_global, k_neg_backbone_feat
    
    
    def featprop_1d(self, feat):
        N, C, T = feat.shape

        # Value transformation
        feat_value = self.value_transform(feat)  # e.g., 1D conv or linear
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, T)

        # Similarity calculation
        feat = F.normalize(feat, dim=1)
        feat = feat.view(N, C, T)

        # Attention: [N, T, T]
        attention = torch.bmm(feat.transpose(1, 2), feat)  # [N, T, T]
        attention = torch.clamp(attention, min=0.)
        # if self.pixpro_p < 1.:
        #     attention = attention + 1e-6
        attention = attention ** 2

        # Propagation: [N, C, T]
        feat = torch.bmm(feat_value, attention.transpose(1, 2))  # [N, C, T]

        return feat


    def regression_loss_1d_fixed(self, q, k, pos_ratio=0.3):
        """
        q, k: [N, C, T=19] - both assumed to be uniformly downsampled from [0,150] and [0,300]
        """
        N, C, T = q.shape

        q = q.view(N, C, T)
        k = k.view(N, C, T)

        # 固定时间范围
        coord_q = torch.tensor([0., 150.], device=q.device).view(1, 2).expand(N, 2)
        coord_k = torch.tensor([0., 300.], device=k.device).view(1, 2).expand(N, 2)

        # 时间 bin 大小
        q_bin_size = ((coord_q[:, 1] - coord_q[:, 0]) / T).view(N, 1, 1)
        k_bin_size = ((coord_k[:, 1] - coord_k[:, 0]) / T).view(N, 1, 1)

        # 计算中心时间
        t_array = torch.arange(T, device=q.device, dtype=torch.float32).view(1, 1, T)
        center_q_t = (t_array + 0.5) * q_bin_size + coord_q[:, 0].view(N, 1, 1)
        center_k_t = (t_array + 0.5) * k_bin_size + coord_k[:, 0].view(N, 1, 1)

        # 距离矩阵
        time_diff = torch.abs(center_q_t.transpose(2, 1) - center_k_t)
        max_bin = torch.max(q_bin_size, k_bin_size)
        norm_dist = time_diff / (max_bin + 1e-6)

        # 正样本 mask
        pos_mask = (norm_dist < pos_ratio).float().detach()

        # 相似度
        q_norm = F.normalize(q, dim=1)
        k_norm = F.normalize(k, dim=1)
        sim = torch.bmm(q_norm.transpose(1, 2), k_norm)

        # 加权平均损失
        loss = (sim * pos_mask).sum(dim=(1, 2)) / (pos_mask.sum(dim=(1, 2)) + 1e-6)
        return -2 * loss.mean()
# norm_dist[0][2,2]


    def forward(self, im_q, im_k=None, im_q_dc=None, nnm=False, topk=1):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        if nnm:
            return self.nearest_neighbors_mining(im_q, im_k, im_q_dc, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_q, im_k, k_neg_A, k_neg_M, k_neg_dense, k_neg_dense_global, k_neg_backbone_feat = self.relative_tempo(im_q, im_k)
            k_A, k_M, k_dense, k_dense_global,k_backbone_feat = self.encoder_k(im_k)  # keys: NxC
            k_A = F.normalize(k_A, dim=1)
            k_M = F.normalize(k_M, dim=1)

            k_neg_dense = F.normalize(k_neg_dense, dim=1)
            k_neg_sampled = k_neg_dense[:, :, torch.randint(k_neg_dense.shape[2], (1,)).item()]
            k_neg_sampled = F.normalize(k_neg_sampled, dim=1)
            k_neg_dense_global = F.normalize(k_neg_dense_global, dim=1)
            
            k_dense = F.normalize(k_dense, dim=1)
            k_dense_global = F.normalize(k_dense_global, dim=1)

        # compute query features
        q_A, q_M,  q_dense, q_dense_global, q_backbone_feat = self.encoder_q(im_q)
        q_A = F.normalize(q_A, dim=1)
        q_M = F.normalize(q_M, dim=1)
        
        q_dense = F.normalize(q_dense, dim=1)
        q_dense_global = F.normalize(q_dense_global, dim=1)

        q_dc = self.encoder_q(im_q_dc,DC=True)
        q_dc = F.normalize(q_dc, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_A1 = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_A2 = torch.einsum('nc,nc->n', [q_A, k_neg_A]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_A = torch.einsum('nc,ck->nk', [q_A, self.queue.clone().detach()])


        l_pos_M = torch.einsum('nc,nc->n', [q_M, k_M]).unsqueeze(-1)
        l_neg_M = torch.einsum('nc,nc->n', [q_M, k_neg_M]).unsqueeze(-1)


        '''start propogate yourself'''
        q_dense_pred = self.featprop_1d(q_dense)
        q_dense_pred = F.normalize(q_dense_pred, dim=1)

        # k_neg_dense_pred = self.featprop_1d(k_neg_dense)
        # k_neg_dense_pred = F.normalize(k_neg_dense_pred, dim=1)

        loss_pro = self.regression_loss_1d_fixed(q_dense_pred,k_neg_dense)
        '''end propogate yourself'''


        '''start densecl'''
        backbone_sim_matrix = torch.matmul(q_backbone_feat.permute(0, 2, 1), k_neg_backbone_feat)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]
        indexed_k_grid = torch.gather(
            k_neg_dense ,  # (B, 128, 16)
            2,
            densecl_sim_ind.unsqueeze(1).expand(-1, k_neg_dense.size(1), -1)  # → (B, 128, 49)
        )
        densecl_sim_q = (q_dense * indexed_k_grid).sum(1)  # → (B, 49)
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)
        q_dense = q_dense.permute(0, 2, 1)  # (B, 19, 128)
        q_dense = q_dense.reshape(-1, q_dense.size(2))  # (B×16, 128)
        l_neg_dense = torch.einsum('nc,ck->nk', [q_dense, self.queue2.clone().detach()])
        
        '''end densecl'''
       

        l_pos_A1_dc = torch.einsum('nc,nc->n', [q_dc, k_A]).unsqueeze(-1)
        l_neg_A_dc = torch.einsum('nc,ck->nk', [q_dc, self.queue.clone().detach()])

        l_pos_A1 /= self.T
        l_pos_A2 /= self.T
        l_neg_A /= self.T
        l_pos_M /= self.T
        l_neg_M /= self.T
        
        # l_pos_A1_dc /= self.T
        l_neg_A_dc /= self.T
        
        l_pos_dense/= self.T
        l_neg_dense/= self.T
        
        # logits: Nx(1+K)
        logits1 = torch.cat([l_pos_A1, l_neg_A], dim=1)
        logits2 = torch.cat([l_pos_A2, l_neg_A], dim=1)
        logits_dense= torch.cat([l_pos_dense, l_neg_dense], dim=1)
        
        logits_M = (l_pos_M, l_neg_M)

        # labels: positive key indicators
        labels_A = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()
        labels_M = torch.ones_like(labels_A)
        labels_dense = torch.zeros(logits_dense.shape[0], dtype=torch.long).cuda()
        
        
        logits_A1_dc = torch.cat([l_pos_A1_dc, l_neg_A_dc], dim=1)
        logits_A1_dc = torch.softmax(logits_A1_dc, dim=1)
        labels_dc = logits1.clone().detach()
        labels_dc = torch.softmax(labels_dc, dim=1)
        labels_dc = labels_dc.detach()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k_neg_A)
        self._dequeue_and_enqueue2(k_neg_dense_global) # 更新 queue2（密集）


        return logits1, logits2, labels_A, logits_M, labels_M, logits_A1_dc, labels_dc, logits_dense, labels_dense,loss_pro
        # return logits1, logits2, labels_A, logits_M, labels_M, 0, 0, logits_dense, labels_dense

    def nearest_neighbors_mining(self, im_q, im_k, im_q_dc, topk=1):

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_q, im_k, k_neg_A, k_neg_M,k_neg_dense, k_neg_dense_global,k_neg_backbone_feat = self.relative_tempo(im_q, im_k)
            k_A, k_M,a,b,c = self.encoder_k(im_k)  # keys: NxC
            k_A = F.normalize(k_A, dim=1)
            k_M = F.normalize(k_M, dim=1)
            
            k_neg_dense = F.normalize(k_neg_dense, dim=1)
            k_neg_sampled = k_neg_dense[:, :, torch.randint(k_neg_dense.shape[2], (1,)).item()]
            k_neg_sampled = F.normalize(k_neg_sampled, dim=1)
            k_neg_dense_global = F.normalize(k_neg_dense_global, dim=1)


        # compute query features
        q_A, q_M,q_dense, q_dense_global, q_backbone_feat = self.encoder_q(im_q)
        q_A = F.normalize(q_A, dim=1)
        q_M = F.normalize(q_M, dim=1)
        q_dense = F.normalize(q_dense, dim=1)
        q_dense_global = F.normalize(q_dense_global, dim=1)
        
        q_dc = self.encoder_q(im_q_dc,DC=True)
        q_dc = F.normalize(q_dc, dim=1)


        '''start propogate yourself'''
        q_dense_pred = self.featprop_1d(q_dense)
        q_dense_pred = F.normalize(q_dense_pred, dim=1)

        # k_neg_dense_pred = self.featprop_1d(k_neg_dense)
        # k_neg_dense_pred = F.normalize(k_neg_dense_pred, dim=1)

        loss_pro = self.regression_loss_1d_fixed(q_dense_pred,k_neg_dense)
        '''end propogate yourself'''

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_A1 = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_A2 = torch.einsum('nc,nc->n', [q_A, k_neg_A]).unsqueeze(-1)
        l_pos_M = torch.einsum('nc,nc->n', [q_M, k_M]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_A = torch.einsum('nc,ck->nk', [q_A, self.queue.clone().detach()])
        l_neg_M = torch.einsum('nc,nc->n', [q_M, k_neg_M]).unsqueeze(-1)

        l_pos_A1_dc = torch.einsum('nc,nc->n', [q_dc, k_A]).unsqueeze(-1)
        l_neg_A_dc = torch.einsum('nc,ck->nk', [q_dc, self.queue.clone().detach()])

        ''''''
        backbone_sim_matrix = torch.matmul(q_backbone_feat.permute(0, 2, 1), k_neg_backbone_feat)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]
        indexed_k_grid = torch.gather(
            k_neg_dense ,  # (B, 128, 19)
            2,
            densecl_sim_ind.unsqueeze(1).expand(-1, k_neg_dense.size(1), -1)  # → (B, 128, 19)
        )
        densecl_sim_q = (q_dense * indexed_k_grid).sum(1)  # → (B, 19)
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) # → (B*19,1)
        q_dense = q_dense.permute(0, 2, 1)  # (B, 19, 128)
        q_dense = q_dense.reshape(-1, q_dense.size(2))  # (B×19, 128)
        l_neg_dense = torch.einsum('nc,ck->nk', [q_dense, self.queue2.clone().detach()])
        
        ''''''
        l_pos_A1 /= self.T
        l_pos_A2 /= self.T
        l_neg_A /= self.T
        l_pos_M /= self.T
        l_neg_M /= self.T

        l_pos_A1_dc /= self.T
        l_neg_A_dc /= self.T
        
        l_pos_dense/= self.T
        l_neg_dense/= self.T

        logits1 = torch.cat([l_pos_A1, l_neg_A], dim=1)
        logits2 = torch.cat([l_pos_A2, l_neg_A], dim=1)
        logits_M = (l_pos_M, l_neg_M)
        logits_dense = torch.cat([l_pos_dense, l_neg_dense], dim=1)
        
        logits_A1_dc = torch.cat([l_pos_A1_dc, l_neg_A_dc], dim=1)
        logits_A1_dc = torch.softmax(logits_A1_dc, dim=1)

        labels_dc = logits1.clone().detach()
        labels_dc = torch.softmax(labels_dc, dim=1)
        labels_dc = labels_dc.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg_A, topk, dim=1)
        _, topkdix_dc = torch.topk(l_neg_A_dc, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg_A)
        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_dc, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)
        labels_M = torch.ones_like(torch.zeros(logits1.shape[0], dtype=torch.long).cuda())

        ''' ''' 
        _, topkdix_dense = torch.topk(l_neg_dense, 5, dim=1)
        
        topk_onehot_dense = torch.zeros_like(l_neg_dense)
        topk_onehot_dense.scatter_(1, topkdix_dense, 1)

        pos_mask_dense = torch.cat([torch.ones(topk_onehot_dense.size(0), 1).cuda(), topk_onehot_dense], dim=1)
        ''' ''' 

        self._dequeue_and_enqueue(k_neg_A)
        self._dequeue_and_enqueue2(k_neg_dense_global) # 更新 queue2（密集）

        return logits1, logits2, pos_mask, logits_M, labels_M, logits_A1_dc, labels_dc, logits_dense,pos_mask_dense,loss_pro
        # return logits1, logits2, pos_mask, logits_M, labels_M, 0, 0, logits_dense, labels_dense
