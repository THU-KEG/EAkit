#!/bin/sh
cd ..
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log distmult \
#                                     --data_dir "data/DBP15K/zh_en" \
#                                     --rate 0.3 \
#                                     --epoch 2000 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 1024 \
#                                     --share \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "DistMult" \
#                                     --sampling "T" \
#                                     --k "5" \
#                                     --margin "1" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --test_dist "cosine"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log distmult \
#                                     --data_dir "data/DBP15K/ja_en" \
#                                     --rate 0.3 \
#                                     --epoch 250 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 1024 \
#                                     --share \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "DistMult" \
#                                     --sampling "T" \
#                                     --k "10" \
#                                     --margin "10" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --test_dist "cosine"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log distmult \
#                                     --data_dir "data/DBP15K/fr_en" \
#                                     --rate 0.3 \
#                                     --epoch 300 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 1024 \
#                                     --share \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "DistMult" \
#                                     --sampling "T" \
#                                     --k "10" \
#                                     --margin "10" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --test_dist "cosine"