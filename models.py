#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    models.py
  Author:       locke
  Date created: 2020/3/25 下午6:59
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *


# --- torch_geometric Packages ---
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_remaining_self_loops
from torch_scatter import scatter_add
# --- torch_geometric Packages end ---




# --- Main Models: Encoder ---
class Encoder(torch.nn.Module):
    def __init__(self, name, hiddens, heads, activation, feat_drop, attn_drop, negative_slope, bias):
        super(Encoder, self).__init__()
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = activation
        self.feat_drop = feat_drop
        for l in range(0, self.num_layers):
            if self.name == "gcn-align":
                self.gnn_layers.append(
                    GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l+1], improved=False, cached=True, bias=bias)
                )
            elif self.name == "naea":
                self.gnn_layers.append(
                    NAEA_GATConv(in_channels=self.hiddens[l]*self.heads[l-1], out_channels=self.hiddens[l+1], heads=self.heads[l], concat=True, negative_slope=negative_slope, dropout=attn_drop, bias=bias)
                )
            elif self.name == "kecg":
                self.gnn_layers.append(
                    KECG_GATConv(in_channels=self.hiddens[l]*self.heads[l-1], out_channels=self.hiddens[l+1], heads=self.heads[l], concat=False, negative_slope=negative_slope, dropout=attn_drop, bias=bias)
                )
            # elif self.name == "SLEF-DESIGN":
            #     self.gnn_layers.append(
            #         SLEF-DESIGN_Conv()
            #     )
            else:
                raise NotImplementedError("bad encoder name: " + self.name)
        if self.name == "naea":
            self.weight = Parameter(torch.Tensor(self.hiddens[0], self.hiddens[-1]))
            nn.init.xavier_normal_(self.weight)
        # if self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: extra parameters'''

    def forward(self, edges, x, r=None):
        edges = edges.t()
        if self.name == "alinet":
            stack = [F.normalize(x, p=2, dim=1)]
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges)
                stack.append(F.normalize(x_, p=2, dim=1))
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            return torch.cat(stack, dim=1)
        elif self.name == "naea":
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges, r)
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            x = torch.sigmoid(x)
            return (x, self.weight)
        # elif self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: special encoder forward'''
        else:
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges)
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            return x            

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, "\n".join([layer.__repr__() for layer in self.gnn_layers]))
# --- Main Models: Encoder end ---



# --- Main Models: Decoder ---
class Decoder(torch.nn.Module):
    def __init__(self, name, params):
        super(Decoder, self).__init__()
        self.print_name = name
        if name.startswith("[") and name.endswith("]"):
            self.name = name[1:-1]
        else:
            self.name = name

        p = 1 if params["train_dist"] == "manhattan" else 2
        transe_sp = True if params["train_dist"] == "normalize_manhattan" else False
        self.feat_drop = params["feat_drop"]
        self.k = params["k"]
        self.alpha = params["alpha"]
        self.margin = params["margin"]
        self.boot = params["boot"]

        if self.name == "align":
            self.func = Align(p)
        elif self.name == "n_transe":
            self.func = N_TransE(p=p, params=self.margin)
        elif self.name == "n_r_align":
            self.func = N_R_Align(params=self.margin)
        elif self.name == "mtranse_align":
            self.func = MTransE_Align(p=p, dim=params["dim"], mode="sa4")
        elif self.name == "alignea":
            self.func = AlignEA(p=p, feat_drop=self.feat_drop, params=self.margin)
        elif self.name == "transedge":
            self.func = TransEdge(p=p, feat_drop=self.feat_drop, dim=params["dim"], mode="cp", params=self.margin)
        elif self.name == "mmea":
            self.func = MMEA(feat_drop=self.feat_drop)
        elif self.name == "transe":
            self.func = TransE(p=p, feat_drop=self.feat_drop, transe_sp=transe_sp)
        elif self.name == "transh":
            self.func = TransH(p=p, feat_drop=self.feat_drop)
        elif self.name == "transr":
            self.func = TransR(p=p, feat_drop=self.feat_drop)
        elif self.name == "distmult":
            self.func = DistMult(feat_drop=self.feat_drop)
        elif self.name == "complex":
            self.func = ComplEx(feat_drop=self.feat_drop)
        elif self.name == "rotate":
            self.func = RotatE(p=p, feat_drop=self.feat_drop, dim=params["dim"], params=self.margin)
        elif self.name == "hake":
            self.func = HAKE(p=p, feat_drop=self.feat_drop, dim=params["dim"], params=self.margin)
        elif self.name == "conve":
            self.func = ConvE(feat_drop=self.feat_drop, dim=params["dim"], e_num=params["e_num"])
        # elif self.name == "SLEF-DESIGN":
            # self.func = SLEF-DESIGN()
        else:
            raise NotImplementedError("bad decoder name: " + self.name)
        
        if params["sampling"] == "T":
            # self.sampling_method = multi_typed_sampling
            self.sampling_method = typed_sampling
        elif params["sampling"] == "N":
            self.sampling_method = nearest_neighbor_sampling
        elif params["sampling"] == "R":
            self.sampling_method = random_sampling
        elif params["sampling"] == ".":
            self.sampling_method = None
        # elif params["sampling"] == "SLEF-DESIGN":
        #     self.sampling_method = SLEF-DESIGN_sampling
        else:
            raise NotImplementedError("bad sampling method: " + self.sampling_method)

        if hasattr(self.func, "loss"):
            self.loss = self.func.loss
        else:
            self.loss = nn.MarginRankingLoss(margin=self.margin)
        if hasattr(self.func, "mapping"):
            self.mapping = self.func.mapping

    def forward(self, ins_emb, rel_emb, sample):
        if type(ins_emb) == tuple:
            ins_emb, weight = ins_emb
            rel_emb_ = torch.matmul(rel_emb, weight)
        else:
            rel_emb_ = rel_emb
        func = self.func if self.sampling_method else self.func.only_pos_loss
        if self.name in ["align", "mtranse_align"]:
            return func(ins_emb[sample[:, 0]], ins_emb[sample[:, 1]])
        elif self.name == "n_r_align":
            nei_emb, ins_emb = ins_emb, rel_emb
            return func(ins_emb[sample[:, 0]], ins_emb[sample[:, 1]], nei_emb[sample[:, 0]], nei_emb[sample[:, 1]])
        # elif self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: special decoder forward'''
        else:
            return func(ins_emb[sample[:, 0]], rel_emb_[sample[:, 1]], ins_emb[sample[:, 2]])

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.print_name, self.func.__repr__())
# --- Main Models: Decoder end ---



# ---   ---   ---   ---   ---   ---
# ---   ---   ---   ---   ---   ---



# --- Decoding Modules ---
class Align(torch.nn.Module):
    def __init__(self, p):
        super(Align, self).__init__()
        self.p = p

    def forward(self, e1, e2):
        pred = - torch.norm(e1 - e2, p=self.p, dim=1)
        return pred

    def only_pos_loss(self, e1, r, e2):
        return - (F.logsigmoid(- torch.sum(torch.pow(e1 + r - e2, 2), 1))).sum()


class N_TransE(torch.nn.Module):
    def __init__(self, p, params):
        super(N_TransE, self).__init__()
        self.p = p
        self.params = params  # mu_1, gamma, mu_2, beta

    def forward(self, e1, r, e2):
        pred = - torch.norm(e1 + r - e2, p=self.p, dim=1)
        return pred
    
    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score + self.params[0] - neg_score).sum() + self.params[1] * F.relu(pos_score - self.params[2]).sum()

class N_R_Align(torch.nn.Module):
    def __init__(self, params):
        super(N_R_Align, self).__init__()
        self.params = params  # beta
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, e1, e2, n1, n2):
        return self.params * torch.sigmoid(self.cos_sim(n1, n2)) + (1 - self.params) * torch.sigmoid(self.cos_sim(e1, e2))

    def loss(self, pos_score, neg_score, target):
        return - torch.log(pos_score).sum()


class MTransE_Align(torch.nn.Module):
    def __init__(self, p, dim, mode="sa4"):
        super(MTransE_Align, self).__init__()
        self.p = p
        self.mode = mode
        if self.mode == "sa1":
            pass
        elif self.mode == "sa3":
            self.weight = Parameter(torch.Tensor(dim))
            nn.init.xavier_normal_(self.weight)
        elif self.mode == "sa4":
            self.weight = Parameter(torch.Tensor(dim, dim))
            nn.init.orthogonal_(self.weight)
            self.I = Parameter(torch.eye(dim), requires_grad=False)
        else:
            raise NotImplementedError

    def forward(self, e1, e2):
        if self.mode == "sa1":
            pred = - torch.norm(e1 - e2, p=self.p, dim=1)
        elif self.mode == "sa3":
            pred = - torch.norm(e1 + self.weight - e2, p=self.p, dim=1)
        elif self.mode == "sa4":
            pred = - torch.norm(torch.matmul(e1, self.weight) - e2, p=self.p, dim=1)
        else:
            raise NotImplementedError
        return pred
    
    def mapping(self, emb):
        return torch.matmul(emb, self.weight)

    def only_pos_loss(self, e1, e2):
        if self.p == 1:
            map_loss = torch.sum(torch.abs(torch.matmul(e1, self.weight) - e2), dim=1).sum()
        else:
            map_loss = torch.sum(torch.pow(torch.matmul(e1, self.weight) - e2, 2), dim=1).sum()
        orthogonal_loss = torch.pow(torch.matmul(self.weight, self.weight.t()) - self.I, 2).sum(dim=1).sum(dim=0)
        return map_loss + orthogonal_loss

    def __repr__(self):
        return '{}(mode={})'.format(self.__class__.__name__, self.mode)


class AlignEA(torch.nn.Module):
    def __init__(self, p, feat_drop, params):
        super(AlignEA, self).__init__()
        self.params = params  # gamma_1, mu_1, gamma_2

    def forward(self, e1, r, e2):
        return torch.sum(torch.pow(e1 + r - e2, 2), 1)
    
    def only_pos_loss(self, e1, r, e2):
        return - (F.logsigmoid(- torch.sum(torch.pow(e1 + r - e2, 2), 1))).sum()

    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score - self.params[0]).sum() + self.params[1] * F.relu(self.params[2] - neg_score).sum()


class TransEdge(torch.nn.Module):
    def __init__(self, p, feat_drop, dim, mode, params):
        super(TransEdge, self).__init__()
        self.func = TransE(p, feat_drop)
        self.params = params  # gamma_1, alpha, gamma_2
        self.mode = mode
        if self.mode == "cc":
            self.mlp_1 = MLP(act=torch.tanh, hiddens=[2*dim, dim], l2_norm=True)
            self.mlp_2 = MLP(act=torch.tanh, hiddens=[2*dim, dim], l2_norm=True)
            self.mlp_3 = MLP(act=torch.tanh, hiddens=[2*dim, dim], l2_norm=True)
        elif self.mode == "cp":
            self.mlp = MLP(act=torch.tanh, hiddens=[2*dim, dim], l2_norm=False)
        else:
            raise NotImplementedError

    def forward(self, e1, r, e2):
        if self.mode == "cc":
            hr = torch.cat((e1, r), dim=1)
            rt = torch.cat((r, e2), dim=1)
            hr = F.normalize(self.mlp_2(hr), p=2, dim=1)
            rt = F.normalize(self.mlp_3(rt), p=2, dim=1)
            crs = F.normalize(torch.cat((hr, rt), dim=1), p=2, dim=1)
            psi = self.mlp_1(crs)
        elif self.mode == "cp":
            ht = torch.cat((e1, e2), dim=1)
            ht = F.normalize(self.mlp(ht), p=2, dim=1)
            psi = r - torch.sum(r * ht, dim=1, keepdim=True) * ht
        else:
            raise NotImplementedError
        psi = torch.tanh(psi)
        return - self.func(e1, psi, e2)
    
    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score - self.params[0]).sum() + self.params[1] * F.relu(self.params[2] - neg_score).sum()


class DistMA(torch.nn.Module):
    def __init__(self, feat_drop):
        super(DistMA, self).__init__()
        self.feat_drop = feat_drop

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        return (e1 * r + e1 * e2 + r * e2).sum(dim=1)

class ComplEx(torch.nn.Module):
    def __init__(self, feat_drop):
        super(ComplEx, self).__init__()
        self.feat_drop = feat_drop

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        e1_r, e1_i = torch.chunk(e1, 2, dim=1)
        r_r, r_i = torch.chunk(r, 2, dim=1)
        e2_r, e2_i = torch.chunk(e2, 2, dim=1)
        return (e1_r * r_r * e2_r + \
                e1_r * r_i * e2_i + \
                e1_i * r_r * e2_i - \
                e1_i * r_i * e2_r).sum(dim=1)

class MMEA(torch.nn.Module):
    def __init__(self, feat_drop):
        super(MMEA, self).__init__()
        self.distma = DistMA(feat_drop)
        self.complex = ComplEx(feat_drop)

    def forward(self, e1, r, e2):
        e1_1, e1_2 = torch.chunk(e1, 2, dim=1)
        r_1, r_2 = torch.chunk(r, 2, dim=1)
        e2_1, e2_2 = torch.chunk(e2, 2, dim=1)
        E1 = self.distma(e1_1, r_1, e2_1)
        E2 = self.complex(e1_2, r_2, e2_2)
        E = E1 + E2
        return torch.cat((E1.view(-1, 1), E2.view(-1, 1), E.view(-1, 1)), dim=1)
    
    def loss(self, pos_score, neg_score, target):
        E1_p_s, E2_p_s, E_p_s = torch.chunk(pos_score, 3, dim=1)
        E1_n_s, E2_n_s, E_n_s = torch.chunk(neg_score, 3, dim=1)
        return - F.logsigmoid(E1_p_s).sum() - F.logsigmoid(-1.0 * E1_n_s).sum() \
                - F.logsigmoid(E2_p_s).sum() - F.logsigmoid(-1.0 * E2_n_s).sum() \
                - F.logsigmoid(E_p_s).sum() - F.logsigmoid(-1.0 * E_n_s).sum()


class TransE(torch.nn.Module):
    def __init__(self, p, feat_drop, transe_sp=False):
        super(TransE, self).__init__()
        self.p = p
        self.feat_drop = feat_drop
        self.transe_sp = transe_sp

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        if self.transe_sp:
            pred = - F.normalize(e1 + r - e2, p=2, dim=1).sum(dim=1)
        else:
            pred = - torch.norm(e1 + r - e2, p=self.p, dim=1)    
        return pred
    
    def only_pos_loss(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        if self.p == 1:
            return torch.sum(torch.abs(e1 + r - e2), dim=1).sum()
        else:
            return torch.sum(torch.pow(e1 + r - e2, 2), dim=1).sum()


class TransH(torch.nn.Module):
    def __init__(self, p, feat_drop, l2_norm=True):
        super(TransH, self).__init__()
        self.p = p
        self.feat_drop = feat_drop
        self.l2_norm = l2_norm

    def forward(self, e1, r, e2):
        if self.l2_norm:
            e1 = F.normalize(e1, p=2, dim=1)
            e2 = F.normalize(e2, p=2, dim=1)
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        d_r, n_r = torch.chunk(r, 2, dim=1)
        if self.l2_norm:
            d_r = F.normalize(d_r, p=2, dim=1)
            n_r = F.normalize(n_r, p=2, dim=1)
        e1_ = e1 - torch.sum(e1 * n_r, dim=1, keepdim=True) * n_r
        e2_ = e2 - torch.sum(e2 * n_r, dim=1, keepdim=True) * n_r
        pred = - torch.norm(e1_ + d_r - e2_, p=self.p, dim=1)
        return pred


class TransR(torch.nn.Module):
    def __init__(self, p, feat_drop):
        super(TransR, self).__init__()
        self.p = p
        self.feat_drop = feat_drop

    def forward(self, e1, rM, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        r, M_r = rM[:, :e1.size(1)], rM[:, e1.size(1):]
        M_r = M_r.view(e1.size(0), e1.size(1), e1.size(1))
        hr = torch.matmul(e1.view(e1.size(0), 1, e1.size(1)), M_r).view(e1.size(0), -1)
        tr = torch.matmul(e2.view(e2.size(0), 1, e2.size(1)), M_r).view(e2.size(0), -1)
        hr = F.normalize(hr, p=2, dim=1)
        tr = F.normalize(tr, p=2, dim=1)
        pred = - torch.norm(hr + r - tr, p=self.p, dim=1)
        return pred


class DistMult(torch.nn.Module):
    def __init__(self, feat_drop):
        super(DistMult, self).__init__()
        self.feat_drop = feat_drop

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        pred = torch.sum(e1 * r * e2, dim=1)
        return pred
    
    # def loss(self, pos_score, neg_score, target):
    #     return F.softplus(-pos_score).sum() + F.softplus(neg_score).sum()


class RotatE(torch.nn.Module):
    def __init__(self, p, feat_drop, dim, params=None):
        super(RotatE, self).__init__()
        # self.p = p
        self.feat_drop = feat_drop
        self.margin = params
        self.rel_range = (self.margin + 2.0) / (dim / 2)
        self.pi = 3.14159265358979323846

    def forward(self, e1, r, e2):    
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        re_head, im_head = torch.chunk(e1, 2, dim=1)
        re_tail, im_tail = torch.chunk(e2, 2, dim=1)
        r = r / (self.rel_range / self.pi)
        re_relation = torch.cos(r)
        im_relation = torch.sin(r)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        pred = score.norm(dim=0).sum(dim=-1)
        return pred
    
    def loss(self, pos_score, neg_score, target):
        return - (F.logsigmoid(self.margin - pos_score) + F.logsigmoid(neg_score - self.margin)).mean()


class HAKE(torch.nn.Module):
    def __init__(self, p, feat_drop, dim, params=None):
        super(HAKE, self).__init__()
        # self.p = p
        self.feat_drop = feat_drop
        self.margin = params
        self.rel_range = (self.margin + 2.0) / (dim / 2)
        self.pi = 3.14159265358979323846
        self.modulus_weight = nn.Parameter(torch.Tensor([1.0]))
        self.phase_weight = nn.Parameter(torch.Tensor([0.5 * self.rel_range]))

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        phase_head, mod_head = torch.chunk(e1, 2, dim=1)
        phase_relation, mod_relation, bias_relation = torch.chunk(r, 3, dim=1)
        phase_tail, mod_tail = torch.chunk(e2, 2, dim=1)
        phase_head = phase_head / (self.rel_range / self.pi)
        phase_relation = phase_relation / (self.rel_range / self.pi)
        phase_tail = phase_tail / (self.rel_range / self.pi)
        phase_score = phase_head + phase_relation - phase_tail
        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]
        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=1) * self.phase_weight
        r_score = torch.norm(r_score, dim=1) * self.modulus_weight
        return (phase_score + r_score)
    
    def loss(self, pos_score, neg_score, target):
        return - (F.logsigmoid(self.margin - pos_score) + F.logsigmoid(neg_score - self.margin)).mean()


class ConvE(torch.nn.Module):
    def __init__(self, feat_drop, dim, e_num):
        super(ConvE, self).__init__()
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.emb_dim1 = 10
        self.emb_dim2 = dim // self.emb_dim1
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)  # in_
        self.bn1 = torch.nn.BatchNorm2d(32) # out
        self.bn2 = torch.nn.BatchNorm1d(dim)
        # self.register_parameter('b', Parameter(torch.zeros(e_num)))
        self.fc = torch.nn.Linear(4608, dim)    # RuntimeError:

    def forward(self, e1, r, e2):
        e1 = e1.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r = r.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1, r], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = torch.mm(x, e2.transpose(1, 0))
        x = (x * e2).sum(dim=1)
        # x += self.b.expand_as(x)
        pred = x
        return pred

    def loss(self, pos_score, neg_score, target):
        return F.binary_cross_entropy_with_logits(torch.cat((pos_score, neg_score), dim=0), torch.cat((target, 1 - target), dim=0))


# class SELF-DESIGN(torch.nn.Module):
#     '''SELF-DESIGN: implement __init__, forward#1 or forward#2, loss(if self-design)'''
#     def __init__(self):
#     def forward(self, e1, r, e2):   # 1
#     def forward(self, e1, e2):      # 2
#     def loss(self, pos_score, neg_score, target):

# --- Decoding Modules end ---



# ---   ---   ---   ---   ---   ---
# ---   ---   ---   ---   ---   ---



# --- Encoding Modules ---
class GCNAlign_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNAlign_GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.mul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class NAEA_GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.weight_2 = Parameter(
            torch.Tensor(in_channels * 2, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.weight_2)
        nn.init.xavier_normal_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, size=None, r_ij=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x, r_ij=r_ij)

    def message(self, edge_index_i, x_i, x_j, size_i, r_ij):
        x_i = torch.matmul(x_i, self.weight)
        x_j = torch.matmul(torch.cat([x_j, r_ij], dim=-1), self.weight_2)

        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class KECG_GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(KECG_GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(1, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.xavier_normal_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = torch.mul(x.repeat((1, self.heads)), self.weight)

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


# class SELF-DESIGN_Conv(MessagePassing):
    # '''SELF-DESIGN: copy code from "Utils: torch_geometric Template" and then modify it'''

# --- Encoding Modules end ---


# ---   ---   ---   ---   ---   ---
# ---   ---   ---   ---   ---   ---


# --- Utils: other Networks ---
class HighWay(torch.nn.Module):
    def __init__(self, f_in, f_out, bias=True):
        super(HighWay, self).__init__()
        self.w = Parameter(torch.Tensor(f_in, f_out))
        nn.init.xavier_uniform_(self.w)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, in_1, in_2):
        t = torch.mm(in_1, self.w)
        if self.bias is not None:
            t = t + self.bias
        gate = torch.sigmoid(t)
        return gate * in_2 + (1.0 - gate) * in_1


class MLP(torch.nn.Module):
    def __init__(self, act=torch.relu, hiddens=[], l2_norm=False):
        super(MLP,self).__init__()
        self.hiddens = hiddens
        self.fc_layers = nn.ModuleList()
        self.num_layers = len(self.hiddens) - 1
        self.activation = act
        self.l2_norm = l2_norm
        for i in range(self.num_layers):
            self.fc_layers.append(nn.Linear(self.hiddens[i], self.hiddens[i+1]))

    def forward(self, e):
        for i, fc in enumerate(self.fc_layers):
            if self.l2_norm:
                e = F.normalize(e, p=2, dim=1)
            e = fc(e)
            if i != self.num_layers-1:
                e = self.activation(e)
        return e
# --- Utils: other Networks end ---



# ---   ---   ---   ---   ---   ---
# ---   ---   ---   ---   ---   ---



# --- Utils: torch_geometric GAT/GCNConv Template ---
class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
# --- Utils: torch_geometric Template end ---
