#!/bin/sh
cd ..
# output/zh_en_transe_test_forR_20200412-0745
# output/ja_en_transe_test_forR_20200412-0755
# output/fr_en_transe_test_20200420-0731

# CUDA_VISIBLE_DEVICES=0 python3 run.py --log transr \
#                                     --data_dir "data/DBP15K/zh_en" \
#                                     --pre "output/zh_en_transe_test_forR_20200412-0745" \
#                                     --rate 0.3 \
#                                     --epoch 200 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 5000 \
#                                     --share \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "TransR" \
#                                     --sampling "T" \
#                                     --k "5" \
#                                     --margin "0.5" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log transr \
#                                     --data_dir "data/DBP15K/ja_en" \
#                                     --pre "output/ja_en_transe_test_forR_20200412-0755" \
#                                     --rate 0.3 \
#                                     --epoch 200 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 5000 \
#                                     --share \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "TransR" \
#                                     --sampling "T" \
#                                     --k "5" \
#                                     --margin "0.5" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"
# CUDA_VISIBLE_DEVICES=0 python3 run.py --log transr \
#                                     --data_dir "data/DBP15K/fr_en" \
#                                     --pre "output/fr_en_transe_test_20200420-0731" \
#                                     --rate 0.3 \
#                                     --epoch 200 \
#                                     --check 10 \
#                                     --update 10 \
#                                     --train_batch_size 5000 \
#                                     --share \
#                                     --encoder "" \
#                                     --hiddens "100" \
#                                     --decoder "TransR" \
#                                     --sampling "T" \
#                                     --k "5" \
#                                     --margin "0.5" \
#                                     --alpha "1" \
#                                     --feat_drop 0.0 \
#                                     --lr 0.01 \
#                                     --train_dist "euclidean" \
#                                     --test_dist "euclidean"