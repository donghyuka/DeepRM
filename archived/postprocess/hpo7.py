import os
import numpy as np
import time
import argparse

args = argparse.ArgumentParser()
args.add_argument("--gpu", type=int, default=0, help="GPU device")
args = args.parse_args()


template = "python -m postprocess.train_site_d20_single --eval_interval 50 --es_patience 10 --es_start 10 --gpu 1 --gpu_pool {} --model {} --batch {} --num_err_layers {} --num_output_layers {} --num_meta_layers {} --num_pred_layers {} --weight_decay {} --dropout {} --lr {} --hidden_dim {} --kernel_size {} --loss {}"


while True:
    time.sleep(1)
    model = np.random.choice(["cnn_model_v1", "cnn_model_v3", "cnn_model_v4", "cnn_model_v5", "cnn_model_v6"])
    batch_size = np.random.choice([16, 32, 64, 128])
    num_err_layers = np.random.choice([0, 1, 2, 4, 8])
    num_meta_layers = np.random.choice([0, 1, 2, 4, 8])
    num_pred_layers = np.random.choice([0, 1, 2, 4, 8])
    num_output_layers = np.random.choice([0, 1, 2, 4, 8, 16])
    weight_decay = np.random.choice([0.3, 0.1, 0.03, 0.01])
    dropout = np.random.choice([0.1, 0.3, 0.5, 0.7])
    lr = np.random.choice([3e-3, 1e-3, 3e-4, 1e-4])
    hidden_dim = np.random.choice([16, 32, 64, 128])
    kernel_size = np.random.choice([3, 5])
    loss = np.random.choice(["MSE", "MAE", "SmoothL1", "Fuchsia", "Iris", "Brassica", "Dracena", "Echeveria", "Freesia"])
    cmd = template.format(args.gpu, model, batch_size, num_err_layers, num_output_layers, num_meta_layers, num_pred_layers, weight_decay, dropout, lr, hidden_dim, kernel_size, loss)
    print(cmd)
    ## Redirect all outputs to /dev/null
    cmd = cmd + " > /dev/null 2>&1"
    os.system(cmd)