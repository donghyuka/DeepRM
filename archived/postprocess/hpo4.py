import os
import numpy as np
import time
import argparse

args = argparse.ArgumentParser()
args.add_argument("--gpu", type=int, default=0, help="GPU device")
args = args.parse_args()


template = "python -m postprocess.train_site_d20_single --eval_interval 50 --es_patience 10 --es_start 10 --gpu 1 --gpu_pool {} --batch {} --num_err_layers {} --num_output_layers {} --num_meta_layers {} --num_pred_layers {} --weight_decay {} --dropout {} --lr {} --hidden_dim {} --kernel_size {} --loss {}"


while True:
    time.sleep(1)
    batch_size = np.random.choice([16, 32, 64, 128])
    num_err_layers = np.random.choice([1, 2, 4, 8])
    num_output_layers = np.random.choice([1, 2, 4, 8])
    num_meta_layers = np.random.choice([1, 2, 4, 8])
    num_pred_layers = np.random.choice([1, 2, 4, 8])
    weight_decay = np.random.choice([2.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.001])
    dropout = np.random.choice([0.0, 0.1, 0.3, 0.5, 0.7])
    lr = np.random.choice([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    hidden_dim = np.random.choice([16, 32, 64, 128, 256])
    kernel_size = np.random.choice([3, 5])
    loss = np.random.choice(["MSE", "MAE", "SmoothL1", "Huber", "Fuchsia", "Cyclamen", "Foxglove", "Hydrangea", "Iris", "Sage"])
    cmd = template.format(args.gpu, batch_size, num_err_layers, num_output_layers, num_meta_layers, num_pred_layers, weight_decay, dropout, lr, hidden_dim, kernel_size, loss)
    print(cmd)
    ## Redirect all outputs to /dev/null
    cmd = cmd + " > /dev/null 2>&1"
    os.system(cmd)


