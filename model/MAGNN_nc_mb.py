import time

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from model.base_MAGNN import MAGNN_ctr_ntype_specific
from RGCN.model import RelationalGraphConvModel


# support for mini-batched forward
# only support one layer for one ctr_ntype
class MAGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.ctr_ntype_layer = MAGNN_ctr_ntype_specific(num_metapaths,
                                                        etypes_list,
                                                        in_dim,
                                                        num_heads,
                                                        attn_vec_dim,
                                                        rnn_type,
                                                        r_vec,
                                                        attn_drop,
                                                        use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        # ctr_ntype-specific layers
        h = self.ctr_ntype_layer(inputs)

        #对应文中（7）式
        h_fc = self.fc(h)
        return h_fc, h


class MAGNN_nc_mb(nn.Module):
    def __init__(self,
                 dataset,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 num_rel,
                 adj_full,
                 A,
                 X,
                 RGCN_argument,
                 feats_opt,
                 gamma,
                 RGCN_layers,
                     emb_alpha=0.5,
                     k_KNN=5,
                     ratio_mask=0.7,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MAGNN_nc_mb, self).__init__()
        self.hidden_dim = hidden_dim
        self.A=A
        self.X=X
        self.feats_opt = feats_opt
        self.norm_adj_full = scipy_csr_to_torch_sparse_tensor(normalize_adj(adj_full))


        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        self.RGCN = RelationalGraphConvModel(
            trans_size=RGCN_argument[0],
            hidden_size=RGCN_argument[1],
            output_size=RGCN_argument[2],
            feat_size_list=feats_dim_list,
            num_bases=RGCN_argument[3],
            num_rel=num_rel,
            num_layer=RGCN_layers,
            num_layer_str=2,
            dropout=RGCN_argument[4],
            adj_full=adj_full,
            norm_adj_full=self.norm_adj_full,
            dataset = dataset,
                gamma = gamma,
                emb_alpha=emb_alpha,
                k_KNN=k_KNN,
                ratio_mask=ratio_mask,
            featureless=False,
            cuda=RGCN_argument[5],
        )
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_nc_mb layers
        self.layer1 = MAGNN_nc_mb_layer(num_metapaths,
                                        num_edge_type,
                                        etypes_list,
                                        hidden_dim,
                                        out_dim,
                                        num_heads,
                                        attn_vec_dim,
                                        rnn_type,
                                        attn_drop=dropout_rate)



    def forward(self, inputs):
        g_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs

        '''encoder-decoder'''
        recon_feat_list, emb_loss, recon_loss = self.RGCN(A=self.A, X=self.X, norm_adj_full=self.norm_adj_full)


        '''# ntype-specific transformation'''
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=recon_feat_list[0].device)
        for i,opt in enumerate(self.feats_opt):
            node_indices = np.where(type_mask == i)[0]
            if opt==1:
                transformed_features[node_indices] = self.fc_list[i](recon_feat_list[i])
            else:
                transformed_features[node_indices] = self.fc_list[i](self.X[i].to_dense())

        #
        h = self.feat_drop(transformed_features)

        '''# hidden layers'''
        logits, h = self.layer1((g_list, h, type_mask, edge_metapath_indices_list, target_idx_list))

        '''for l in range(self.num_layers - 1):
            h, _ = self.layers[l]((g_list, h, type_mask, edge_metapath_indices_list))
            h = F.elu(h)
        # output projection layer
        logits, h = self.layers[-1]((g_list, h, type_mask, edge_metapath_indices_list))'''


        return logits, h, emb_loss,recon_loss,recon_feat_list


def scipy_csr_to_torch_sparse_tensor(csr_mat, device='cuda:0'):
    coo = csr_mat.tocoo()
    indices = np.vstack((coo.row, coo.col))
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(coo.data)
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, size=shape, device=device)

def normalize_adj(adj, symmetric=True):
    """归一化稀疏邻接矩阵"""
    # 加单位矩阵（自环）
    adj = adj + sp.eye(adj.shape[0])

    # 计算度矩阵
    rowsum = np.array(adj.sum(1)).flatten()

    if symmetric:
        # 对称归一化 D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt
    else:
        # 行归一化 D^{-1} A
        d_inv = np.power(rowsum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        D_inv = sp.diags(d_inv)
        return D_inv @ adj