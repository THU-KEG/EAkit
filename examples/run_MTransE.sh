#!/bin/sh
cd ..
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log mtranse \
#                                     --data_dir "data/DBP15K/zh_en" \
#                                     --rate 0.3 \
#                                     --epoch 400 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 5000 \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "TransE,MTransE_Align" \
#                                     --sampling ".,." \
#                                     --k "0,0" \
#                                     --margin "0,0" \
#                                     --alpha "1,50" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log mtranse \
#                                     --data_dir "data/DBP15K/ja_en" \
#                                     --rate 0.3 \
#                                     --epoch 400 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 5000 \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "TransE,MTransE_Align" \
#                                     --sampling ".,." \
#                                     --k "0,0" \
#                                     --margin "0,0" \
#                                     --alpha "1,50" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log mtranse \
#                                     --data_dir "data/DBP15K/fr_en" \
#                                     --rate 0.3 \
#                                     --epoch 500 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 5000 \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "TransE,MTransE_Align" \
#                                     --sampling ".,." \
#                                     --k "0,0" \
#                                     --margin "0,0" \
#                                     --alpha "1,50" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"