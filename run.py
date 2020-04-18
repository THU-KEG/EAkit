#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    run.py
  Author:       locke
  Date created: 2020/3/25 下午6:58
"""

import time
import argparse
import os
import gc
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from load_data import *
from models import *
from utils import *
from semi_utils import bootstrapping, boot_update_triple

from torch.utils.tensorboard import SummaryWriter
import logging



class Experiment:
    def __init__(self, args):
        self.save = args.save
        self.save_prefix = "%s_%s" % (args.data_dir.split("/")[-1], args.log)
        
        self.hiddens = list(map(int, args.hiddens.split(",")))
        self.heads = list(map(int, args.heads.split(",")))
        
        self.args = args
        self.args.encoder = args.encoder.lower()
        self.args.decoder = args.decoder.lower().split(",")
        self.args.sampling = args.sampling.split(",")        
        self.args.k = list(map(int, args.k.split(",")))
        self.args.margin = [float(x) if "-" not in x else list(map(float, x.split("-"))) for x in args.margin.split(",")]
        self.args.alpha = list(map(float, args.alpha.split(",")))
        assert len(self.args.decoder) >= 1
        assert len(self.args.decoder) == len(self.args.sampling) and \
                len(self.args.sampling) == len(self.args.k) and \
                len(self.args.k) == len(self.args.alpha)

        self.cached_sample = {}
        self.best_result = ()


    def evaluate(self, it, test, ins_emb, mapping_emb=None):
        t_test = time.time()
        top_k = [1, 3, 5, 10]
        if mapping_emb is not None:
            logger.info("using mapping")
            left_emb = mapping_emb[test[:, 0]]
        else:
            left_emb = ins_emb[test[:, 0]]
        right_emb = ins_emb[test[:, 1]]

        distance = - sim(left_emb, right_emb, metric=self.args.test_dist, normalize=True, csls_k=self.args.csls)
        if self.args.rerank:
            indices = np.argsort(np.argsort(distance, axis=1), axis=1)
            indices_ = np.argsort(np.argsort(distance.T, axis=1), axis=1)
            distance = indices + indices_.T

        tasks = div_list(np.array(range(len(test))), 10)
        pool = multiprocessing.Pool(processes=len(tasks))
        reses = list()
        for task in tasks:
            reses.append(pool.apply_async(multi_cal_rank, (task, distance[task, :], distance[:, task], top_k, self.args)))
        pool.close()
        pool.join()
        
        acc_l2r, acc_r2l = np.array([0.] * len(top_k)), np.array([0.] * len(top_k))
        mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
        for res in reses:
            (_acc_l2r, _mean_l2r, _mrr_l2r, _acc_r2l, _mean_r2l, _mrr_r2l) = res.get()
            acc_l2r += _acc_l2r
            mean_l2r += _mean_l2r
            mrr_l2r += _mrr_l2r
            acc_r2l += _acc_r2l
            mean_r2l += _mean_r2l
            mrr_r2l += _mrr_r2l
        mean_l2r /= len(test)
        mean_r2l /= len(test)
        mrr_l2r /= len(test)
        mrr_r2l /= len(test)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / len(test), 4)
            acc_r2l[i] = round(acc_r2l[i] / len(test), 4)
        
        logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r.tolist(), mean_l2r, mrr_l2r, time.time() - t_test))
        logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l.tolist(), mean_r2l, mrr_r2l, time.time() - t_test))
        for i, k in enumerate(top_k):
            writer.add_scalar("l2r_HitsAt{}".format(k), acc_l2r[i], it)
            writer.add_scalar("r2l_HitsAt{}".format(k), acc_r2l[i], it)
        writer.add_scalar("l2r_MeanRank", mean_l2r, it)
        writer.add_scalar("l2r_MeanReciprocalRank", mrr_l2r, it)
        writer.add_scalar("r2l_MeanRank", mean_r2l, it)
        writer.add_scalar("r2l_MeanReciprocalRank", mrr_r2l, it)
        return (acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l)


    def init_emb(self):
        e_scale, r_scale = 1, 1
        if not self.args.encoder:
            if self.args.decoder == ["rotate"]:
                r_scale = r_scale / 2
            elif self.args.decoder == ["hake"]:
                r_scale = (r_scale / 2) * 3
            elif self.args.decoder == ["transh"]:
                r_scale = r_scale * 2
            elif self.args.decoder == ["transr"]:
                r_scale = self.hiddens[0] + 1
        self.ins_embeddings = nn.Embedding(d.ins_num, self.hiddens[0] * e_scale).to(device)
        self.rel_embeddings = nn.Embedding(d.rel_num, int(self.hiddens[0] * r_scale)).to(device)
        if self.args.decoder == ["rotate"] or self.args.decoder == ["hake"]:
            ins_range = (self.args.margin[0] + 2.0) / float(self.hiddens[0] * e_scale)
            nn.init.uniform_(tensor=self.ins_embeddings.weight, a=-ins_range, b=ins_range)
            rel_range = (self.args.margin[0] + 2.0) / float(self.hiddens[0] * r_scale)
            nn.init.uniform_(tensor=self.rel_embeddings.weight, a=-rel_range, b=rel_range)
            if self.args.decoder == ["hake"]:
                r_dim = int(self.hiddens[0] / 2)
                nn.init.ones_(tensor=self.rel_embeddings.weight[:, r_dim : 2*r_dim])
                nn.init.zeros_(tensor=self.rel_embeddings.weight[:, 2*r_dim : 3*r_dim])
        else:
            nn.init.xavier_normal_(self.ins_embeddings.weight)
            nn.init.xavier_normal_(self.rel_embeddings.weight)
        if "alignea" in self.args.decoder or "mtranse_align" in self.args.decoder or "transedge" in self.args.decoder:
            self.ins_embeddings.weight.data = F.normalize(self.ins_embeddings.weight, p=2, dim=1)
            self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight, p=2, dim=1)
        elif "transr" in self.args.decoder:
            assert self.args.pre != ""
            self.ins_embeddings.weight.data = torch.from_numpy(np.load(self.args.pre+"_ins.npy")).to(device)
            self.rel_embeddings.weight[:, :self.hiddens[0]].data = torch.from_numpy(np.load(self.args.pre+"_rel.npy")).to(device)
        self.enh_ins_emb = self.ins_embeddings.weight.cpu().detach().numpy()
        self.mapping_ins_emb = None


    def train_and_eval(self):
        self.init_emb()

        graph_encoder = None
        if self.args.encoder:
            graph_encoder = Encoder(self.args.encoder, self.hiddens, self.heads+[1], activation=F.elu, feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop, negative_slope=0.2, bias=False).to(device)
            logger.info(graph_encoder)
        knowledge_decoder = []
        for idx, decoder_name in enumerate(self.args.decoder):
            knowledge_decoder.append(Decoder(decoder_name, params={
                "e_num": d.ins_num,
                "r_num": d.rel_num,
                "dim": self.hiddens[-1],
                "feat_drop": self.args.feat_drop,
                "train_dist": self.args.train_dist,
                "sampling": self.args.sampling[idx],
                "k": self.args.k[idx],
                "margin": self.args.margin[idx],
                "alpha": self.args.alpha[idx],
                "boot": self.args.bootstrap,
                # pass other useful parameters to Decoder
            }).to(device))
        logger.info(knowledge_decoder)

        params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight] + [p for k_d in knowledge_decoder for p in list(k_d.parameters())] + (list(graph_encoder.parameters()) if self.args.encoder else []))
        opt = optim.Adagrad(params, lr=self.args.lr, weight_decay=self.args.wd)
        if self.args.dr:
            scheduler = optim.lr_scheduler.ExponentialLR(opt, self.args.dr)
        logger.info(params)
        logger.info(opt)

        # Train
        logger.info("Start training...")
        for it in range(0, self.args.epoch):

            for idx, k_d in enumerate(knowledge_decoder):
                if (k_d.name == "align" and len(d.ill_train_idx) == 0):
                    continue
                t_ = time.time()
                if k_d.print_name.startswith("["):  # Run Independent Model (only decoder)
                    loss = self.train_1_epoch(it, opt, None, k_d, d.ins_G_edges_idx, d.triple_idx, d.ill_train_idx, [d.kg1_ins_ids, d.kg2_ins_ids], d.boot_triple_idx, d.boot_pair_dix, self.ins_embeddings.weight, self.rel_embeddings.weight)
                else:               # Run Basic Model (encoder - decoder)
                    loss = self.train_1_epoch(it, opt, graph_encoder, k_d, d.ins_G_edges_idx, d.triple_idx, d.ill_train_idx, [d.kg1_ins_ids, d.kg2_ins_ids], d.boot_triple_idx, d.boot_pair_dix, self.ins_embeddings.weight, self.rel_embeddings.weight)
                if hasattr(k_d, "mapping"):
                    self.mapping_ins_emb = k_d.mapping(self.ins_embeddings.weight).cpu().detach().numpy()
                loss_name = "loss_" + k_d.print_name.replace("[", "_").replace("]", "_")
                writer.add_scalar(loss_name, loss, it)
                logger.info("epoch: %d\t%s: %.8f\ttime: %ds" % (it, loss_name, loss, int(time.time()-t_)) )

            if self.args.dr:
                scheduler.step()

            # Evaluate
            if (it + 1) % self.args.check == 0:
                logger.info("Start validating...")
                with torch.no_grad():
                    if graph_encoder and graph_encoder.name == "naea":
                        beta = self.args.margin[-1]
                        emb = beta * self.enh_ins_emb + (1 - beta) * self.ins_embeddings.weight.cpu().detach().numpy()
                    else:
                        emb = self.enh_ins_emb
                    if len(d.ill_val_idx) > 0:
                        result = self.evaluate(it, d.ill_val_idx, emb, self.mapping_ins_emb)
                    else:
                        result = self.evaluate(it, d.ill_test_idx, emb, self.mapping_ins_emb)

                # Early Stop
                if self.args.early and len(self.best_result) != 0 and result[0][0] < self.best_result[0][0]:
                    if len(d.ill_val_idx) > 0:
                        logger.info("Start testing...")
                        self.evaluate(it, d.ill_test_idx, emb, self.mapping_ins_emb)
                    else:
                        logger.info("Early stop, best result:")
                        acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = self.best_result
                        logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} ".format(top_k, acc_l2r, mean_l2r, mrr_l2r))
                        logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} \n".format(top_k, acc_r2l, mean_r2l, mrr_r2l))
                    break
                self.best_result = result

            # Bootstrapping
            if self.args.bootstrap and it >= self.args.start_bp and (it + 1) % self.args.update == 0:
                with torch.no_grad():
                    if graph_encoder and graph_encoder.name == "naea":
                        beta = self.args.margin[-1]
                        emb = beta * self.enh_ins_emb + (1 - beta) * self.ins_embeddings.weight.cpu().detach().numpy()
                    else:
                        emb = self.enh_ins_emb
                d.labeled_alignment, A, B = bootstrapping(ref_sim_mat=sim(emb[d.ill_test_idx[:, 0]], emb[d.ill_test_idx[:, 1]], metric=self.args.test_dist, normalize=True, csls_k=0), 
                                                        ref_ent1=d.ill_test_idx[:, 0].tolist(), 
                                                        ref_ent2=d.ill_test_idx[:, 1].tolist(), 
                                                        labeled_alignment=d.labeled_alignment, th=self.args.threshold, top_k=10, is_edit=False)
                if d.labeled_alignment:
                    d.boot_triple_idx = boot_update_triple(A, B, d.triple_idx)
                    d.boot_pair_dix = [(A[i], B[i])for i in range(len(A))]
                    logger.info("Bootstrapping: + " + str(len(A)) + " ills, " + str(len(d.boot_triple_idx)) + " triples.")

        # Save Embeddings
        if self.save != "":
            if not os.path.exists(self.save):
                os.makedirs(self.save)
            time_str = self.save_prefix + "_" + time.strftime("%Y%m%d-%H%M", time.gmtime())
            if graph_encoder:
                with torch.no_grad():
                    graph_encoder.eval()
                    edges = torch.LongTensor(d.ins_G_edges_idx).to(device)
                    enh_emb = graph_encoder(edges, self.ins_embeddings.weight)
                    np.save(self.save + "/%s_enh_ins.npy" % (time_str), enh_emb.cpu().detach().numpy())
            np.save(self.save + "/%s_ins.npy" % (time_str), self.ins_embeddings.weight.cpu().detach().numpy())
            np.save(self.save + "/%s_rel.npy" % (time_str), self.rel_embeddings.weight.cpu().detach().numpy())
            logger.info("Embeddings saved!")
    

    def train_1_epoch(self, it, opt, encoder, decoder, edges, triples, ills, ids, boot_triples, boot_pairs, ins_emb, rel_emb):
        if encoder:
            encoder.train()
        decoder.train()
        losses = []
        if "pos_"+decoder.print_name not in self.cached_sample or it % self.args.update == 0:
            if decoder.name in ["align", "mtranse_align", "n_r_align"]:
                if decoder.boot:
                    self.cached_sample["pos_"+decoder.print_name] = ills.tolist() + boot_pairs
                else:
                    self.cached_sample["pos_"+decoder.print_name] = ills.tolist()
                self.cached_sample["pos_"+decoder.print_name] = np.array(self.cached_sample["pos_"+decoder.print_name])
            else:
                if decoder.boot:
                    self.cached_sample["pos_"+decoder.print_name] = triples + boot_triples
                else:
                    self.cached_sample["pos_"+decoder.print_name] = triples
            np.random.shuffle(self.cached_sample["pos_"+decoder.print_name])
            # print("train size:", len(self.cached_sample["pos_"+decoder.print_name]))
            
        train = self.cached_sample["pos_"+decoder.print_name]
        if self.args.train_batch_size == -1:
            train_batch_size = len(train)
        else:
            train_batch_size = self.args.train_batch_size
        for i in range(0, len(train), train_batch_size):
            pos_batch = train[i:i+train_batch_size]

            if (decoder.print_name+str(i) not in self.cached_sample or it % self.args.update == 0) and decoder.sampling_method:
                self.cached_sample[decoder.print_name+str(i)] = decoder.sampling_method(pos_batch, triples, ills, ids, decoder.k, params={
                    "emb": self.enh_ins_emb,
                    "metric": self.args.test_dist,
                })
            
            if decoder.sampling_method:
                neg_batch = self.cached_sample[decoder.print_name+str(i)]
    
            opt.zero_grad()
            if decoder.sampling_method:
                neg = torch.LongTensor(neg_batch).to(device)
                if neg.size(0) > len(pos_batch) * decoder.k:
                    pos = torch.LongTensor(pos_batch).repeat(decoder.k * 2, 1).to(device)
                elif hasattr(decoder.func, "loss") and decoder.name not in ["rotate", "hake", "conve", "mmea", "n_transe"]:
                    pos = torch.LongTensor(pos_batch).to(device)
                else:
                    pos = torch.LongTensor(pos_batch).repeat(decoder.k, 1).to(device)
            else:
                pos = torch.LongTensor(pos_batch).to(device)

            if encoder:
                use_edges = torch.LongTensor(edges).to(device)
                enh_emb = encoder.forward(use_edges, ins_emb, rel_emb[d.r_ij_idx] if encoder.name=="naea" else None)
            else:
                enh_emb = ins_emb
            
            self.enh_ins_emb = enh_emb[0].cpu().detach().numpy() if encoder and encoder.name == "naea" else enh_emb.cpu().detach().numpy()
            if decoder.name == "n_r_align":
                rel_emb = ins_emb

            if decoder.sampling_method:
                pos_score = decoder.forward(enh_emb, rel_emb, pos)
                neg_score = decoder.forward(enh_emb, rel_emb, neg)
                target = torch.ones(neg_score.size()).to(device)

                loss = decoder.loss(pos_score, neg_score, target) * decoder.alpha
            else:
                loss = decoder.forward(enh_emb, rel_emb, pos) * decoder.alpha
            
            loss.backward()
            
            opt.step()
            losses.append(loss.item())

        return np.mean(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/DBP15K/zh_en", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
    parser.add_argument("--val", type=float, default=0.0, help="valid set rate")
    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding")
    parser.add_argument("--pre", default="", help="pre-train embedding dir (only use in transr)")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--epoch", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check", type=int, default=5, help="check point")
    parser.add_argument("--update", type=int, default=5, help="number of epoch for updating negtive samples")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early", action="store_true", default=False, help="whether to use early stop")    # Early stop when the Hits@1 score begins to drop on the validation sets, checked every 10 epochs.
    parser.add_argument("--share", action="store_true", default=False, help="whether to share ill emb")
    parser.add_argument("--swap", action="store_true", default=False, help="whether to swap ill in triple")
    
    parser.add_argument("--bootstrap", action="store_true", default=False, help="whether to use bootstrap")
    parser.add_argument("--start_bp", type=int, default=9, help="epoch of starting bootstrapping")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold of bootstrap alignment")

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--hiddens", type=str, default="100,100,100", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--attn_drop", type=float, default=0, help="dropout rate for gat layers")

    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use: . min = 1")
    parser.add_argument("--sampling", type=str, default="N", help="negtive sampling method for each decoder")
    parser.add_argument("--k", type=str, default="25", help="negtive sampling number for each decoder")
    parser.add_argument("--margin", type=str, default="1", help="margin for each margin based ranking loss (or params for other loss function)")
    parser.add_argument("--alpha", type=str, default="1", help="weight for each margin based ranking loss")
    parser.add_argument("--feat_drop", type=float, default=0, help="dropout rate for layers")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")    

    parser.add_argument("--train_dist", type=str, default="euclidean", help="distance function used in train (inner, cosine, euclidean, manhattan)")
    parser.add_argument("--test_dist", type=str, default="euclidean", help="distance function used in test (inner, cosine, euclidean, manhattan)")

    parser.add_argument("--csls", type=int, default=0, help="whether to use csls in test (0 means not using)")
    parser.add_argument("--rerank", action="store_true", default=False, help="whether to use rerank in test")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    writer = SummaryWriter("_runs/%s_%s" % (args.data_dir.split("/")[-1], args.log))
    logger.info(args)

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load Data
    d = AlignmentData(data_dir=args.data_dir, rate=args.rate, share=args.share, swap=args.swap, val=args.val, with_r=args.encoder.lower()=="naea")
    logger.info(d)

    experiment = Experiment(args=args)

    t_total = time.time()
    experiment.train_and_eval()
    logger.info("optimization finished!")
    logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))
