#!/bin/sh
cd ..
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log kecg \
#                                     --data_dir "data/DBP15K/zh_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size -1 \
#                                     --encoder "KECG" \
#                                     --hiddens "100,100,100" \
#                                     --heads "2,2" \
#                                     --decoder "TransE,Align" \
#                                     --sampling "T,N" \
#                                     --k "5,25" \
#                                     --margin "3,3" \
#                                     --alpha "1,1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.005 \
#                                     --train_dist "normalize_manhattan" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log kecg \
#                                     --data_dir "data/DBP15K/ja_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size -1 \
#                                     --encoder "KECG" \
#                                     --hiddens "100,100,100" \
#                                     --heads "2,2" \
#                                     --decoder "TransE,Align" \
#                                     --sampling "T,N" \
#                                     --k "5,25" \
#                                     --margin "3,3" \
#                                     --alpha "1,1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.005 \
#                                     --train_dist "normalize_manhattan" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log kecg \
#                                     --data_dir "data/DBP15K/fr_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size -1 \
#                                     --encoder "KECG" \
#                                     --hiddens "100,100,100" \
#                                     --heads "2,2" \
#                                     --decoder "TransE,Align" \
#                                     --sampling "T,N" \
#                                     --k "5,25" \
#                                     --margin "3,3" \
#                                     --alpha "1,1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.005 \
#                                     --train_dist "normalize_manhattan" \
#                                     --test_dist "euclidean"