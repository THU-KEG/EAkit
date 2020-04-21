#!/bin/sh
cd ..
# TODO: Learning from Holistic Perspective
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log bootea \
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
#                                     --encoder "" \
#                                     --hiddens "75" \
#                                     --decoder "AlignEA" \
#                                     --sampling "T" \
#                                     --k "20" \
#                                     --margin "0.2-0.2-1.0" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.05 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log bootea \
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
#                                     --encoder "" \
#                                     --hiddens "75" \
#                                     --decoder "AlignEA" \
#                                     --sampling "T" \
#                                     --k "20" \
#                                     --margin "0.2-0.2-1.0" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.05 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log bootea \
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
#                                     --encoder "" \
#                                     --hiddens "75" \
#                                     --decoder "AlignEA" \
#                                     --sampling "T" \
#                                     --k "20" \
#                                     --margin "0.2-0.2-1.0" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.05 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"