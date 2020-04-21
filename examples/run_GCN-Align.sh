#!/bin/sh
cd ..
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log gcnalign \
#                                     --data_dir "data/DBP15K/zh_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size -1 \
#                                     --encoder "GCN-Align" \
#                                     --hiddens "100,100,100" \
#                                     --decoder "Align" \
#                                     --sampling "N" \
#                                     --k "25" \
#                                     --margin "1" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.005 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log gcnalign \
#                                     --data_dir "data/DBP15K/ja_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size -1 \
#                                     --encoder "GCN-Align" \
#                                     --hiddens "100,100,100" \
#                                     --decoder "Align" \
#                                     --sampling "N" \
#                                     --k "25" \
#                                     --margin "1" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.005 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log gcnalign \
#                                     --data_dir "data/DBP15K/fr_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size -1 \
#                                     --encoder "GCN-Align" \
#                                     --hiddens "100,100,100" \
#                                     --decoder "Align" \
#                                     --sampling "N" \
#                                     --k "25" \
#                                     --margin "1" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.005 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"