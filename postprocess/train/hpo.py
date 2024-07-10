cmd = "python -m postprocess.train.train --data /extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_070324 \
 --output /extdata4/baeklab/Hyeonseo/m6A/postprocess/model/ --tb /extdata4/baeklab/Hyeonseo/m6A/postprocess/tensorboard/ \
 --batch {bat} --eval_batch 64 --lr 0.0002 --epochs 1000 --es_delta 1e-05 --es_patience 10 --es_start 10 --period 30 \
 --buffer_size 10000 --lr_step 3000 --weight_decay 0.02 --loss Echeveria  --input_height 20 --hidden_dim {dim} --output_dim 1 \
 --num_err_layers {nel} --num_meta_layers {nml} --num_pred_layers {npl} --num_output_layers {nol} --dropout_rate {do} --kernel_size 3 --gpu 1 \
 --model pp_model_v{ver} --gpu_pool {gpu} > /dev/null 2>&1 &"

import os
import numpy as np

for gpu in range(4):
    for i in range(32):
        dim = np.random.choice([8,16,32,64])
        nel = np.random.choice([1,2,4,8,16])
        nml = np.random.choice([1,2,4])
        npl = np.random.choice([1,2,4])
        nol = np.random.choice([1,2,4,8])
        do = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
        ver = np.random.choice([1,2,3,4,5,6,7,8])
        bat = np.random.choice([16,32,64])
        os.system(cmd.format(gpu=gpu, dim=dim, nel=nel, nml=nml, npl=npl, nol=nol, do=do, ver=ver, bat=bat))

