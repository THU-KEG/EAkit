#!/bin/sh
cd ..
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log naea \
#                                     --data_dir "data/DBP15K/zh_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 20000 \
#                                     --swap \
#                                     --bootstrap \
#                                     --start_bp 9 \
#                                     --threshold 0.7 \
#                                     --encoder "NAEA" \
#                                     --hiddens "75" \
#                                     --decoder "[N_TransE],N_TransE,N_R_Align" \
#                                     --sampling "T,T,N" \
#                                     --k "10,10,20" \
#                                     --margin "0.5-2-0.2,0.5-2-0.2,0.8" \
#                                     --alpha "0.2,0.8,1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.05 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "cosine"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log naea \
#                                     --data_dir "data/DBP15K/ja_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 20000 \
#                                     --swap \
#                                     --bootstrap \
#                                     --start_bp 9 \
#                                     --threshold 0.7 \
#                                     --encoder "NAEA" \
#                                     --hiddens "75" \
#                                     --decoder "[N_TransE],N_TransE,N_R_Align" \
#                                     --sampling "T,T,N" \
#                                     --k "10,10,20" \
#                                     --margin "0.5-2-0.2,0.5-2-0.2,0.8" \
#                                     --alpha "0.2,0.8,1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.05 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "cosine"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log naea \
#                                     --data_dir "data/DBP15K/fr_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 20000 \
#                                     --swap \
#                                     --bootstrap \
#                                     --start_bp 9 \
#                                     --threshold 0.7 \
#                                     --encoder "NAEA" \
#                                     --hiddens "75" \
#                                     --decoder "[N_TransE],N_TransE,N_R_Align" \
#                                     --sampling "T,T,N" \
#                                     --k "10,10,20" \
#                                     --margin "0.5-2-0.2,0.5-2-0.2,0.8" \
#                                     --alpha "0.2,0.8,1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.05 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "cosine"