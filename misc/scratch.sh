
# AICA
python -m train.train --batch 600 --data /data/Hyeonseo/m6A/dataset/ver030924/main/ --output /data/Hyeonseo/m6A//model --tb /data/Hyeonseo/m6A/tensorboard --class_ratio 3   --gpu 8 --lr 1e-4 --model transformer_prototype_v16 --eval_interval 500 --save_interval 500 --log_interval 10 --lr_step 500 --model transformer_prototype_v16 --head 12 --enc_layer 12 --enc_dim 768 --lin_layer 3

# XE8545
python -m train.train --batch 1600 --data /data/Hyeonseo/m6A/dataset/ver031624/ --output /data/Hyeonseo/m6A/model --tb /data/Hyeonseo/m6A/tensorboard --class_ratio 1   --gpu 4 --lr 1e-4 --model transformer_prototype_v16 --eval_interval 100 --save_interval 100 --log_interval 10 --lr_step 500 --model transformer_prototype_v16 --enc_dim 512

# DEEP1-BIO C1
python -m train.train --gpu 4 --batch 240 --data /extdata4/baeklab/Hyeonseo/m6A/dataset/ver030924/imbx10/ --class_ratio 4 --eval_interval 1000 --save_interval 1000 --log_interval 10 --model transformer_prototype_v16 --enc_layer 12 --enc_dim 768 --lin_layer 3 --lr_step 100 --model transformer_prototype_v16 --rlrop 1e-2

# Deep1 N06
python -m evaluate.evaluate_sample --batch 4000 --gpu 0 --data /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/eval_data/miclip2_glori_m6ace_drach --model /extdata4/baeklab/Hyeonseo/m6A/inference/model_to_eval_2/

# SNU-BIO C1
python -m train.train --gpu 4 --batch 240 --data /extdata2/baeklab/Hyeonseo/m6A/dataset/ver030924/engx10/ --output /extdata2/baeklab/Hyeonseo/m6A/model --tb /extdata2/baeklab/Hyeonseo/m6A/tensorboard --class_ratio 1 --log_interval 10 --eval_interval 1000 --save_interval 1000 --log_interval 10 --model transformer_prototype_v16 --enc_layer 12 --enc_dim 768 --lin_layer 3 --lr_step 100 --model transformer_prototype_v16 --rlrop 1e-2
