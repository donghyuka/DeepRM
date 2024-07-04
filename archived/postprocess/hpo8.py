import os
import numpy as np
import time
import argparse

args = argparse.ArgumentParser()
args.add_argument("--gpu", type=int, default=0, help="GPU device")
args = args.parse_args()


template = "python -m postprocess.train_site_d20_single --eval_interval 50 --es_patience 10 --es_start 10 --gpu 1 --gpu_pool {} --model {} --batch {} --num_pred_layers {}  --num_output_layers {} --weight_decay {} --dropout {} --lr {} --hidden_dim {} --kernel_size {} --loss {}"


while True:
    time.sleep(1)
    model = np.random.choice(["cnn_model_v7", "cnn_model_v8", "cnn_model_v9"])
    batch_size = np.random.choice([16, 32, 64])
    num_output_layers = np.random.choice([0, 1, 2, 4, 8, 16, 32])
    num_pred_layers = np.random.choice([1, 2, 4, 8, 16, 32])
    weight_decay = np.random.choice([0.4, 0.2, 0.1, 0.02, 0.04, 0.01])
    dropout = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    lr = np.random.choice([4e-3, 2e-3, 1e-3, 4e-4, 2e-4, 1e-4])
    hidden_dim = np.random.choice([16, 32, 64, 128, 256])
    kernel_size = np.random.choice([3, 5])
    loss = np.random.choice(["MAE", "Fuchsia", "Iris", "Sage", "Echeveria"])
    cmd = template.format(args.gpu, model, batch_size, num_pred_layers, num_output_layers, weight_decay, dropout, lr, hidden_dim, kernel_size, loss)
    print(cmd)
    ## Redirect all outputs to /dev/null
    cmd = cmd + " > /dev/null 2>&1"
    os.system(cmd)