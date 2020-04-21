#!/bin/sh
cd ..
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log mmea \
#                                     --data_dir "data/DBP15K/zh_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 20000 \
#                                     --share \
#                                     --rerank \
#                                     --encoder "" \
#                                     --hiddens "300" \
#                                     --decoder "MMEA" \
#                                     --sampling "T" \
#                                     --k "10" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --test_dist "inner"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log mmea \
#                                     --data_dir "data/DBP15K/ja_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 20000 \
#                                     --share \
#                                     --rerank \
#                                     --encoder "" \
#                                     --hiddens "300" \
#                                     --decoder "MMEA" \
#                                     --sampling "T" \
#                                     --k "10" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --test_dist "inner"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log mmea \
#                                     --data_dir "data/DBP15K/fr_en" \
#                                     --rate 0.3 \
#                                     --epoch 1000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 20000 \
#                                     --share \
#                                     --rerank \
#                                     --encoder "" \
#                                     --hiddens "300" \
#                                     --decoder "MMEA" \
#                                     --sampling "T" \
#                                     --k "10" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --test_dist "inner"