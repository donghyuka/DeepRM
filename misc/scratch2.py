import torch
import numpy
import time
import os

def test_torch():
    path = "/extdata4/baeklab/Hyeonseo/m6A/dataset/ver082724_tensor/score-perfect/train/pos/"
    files = os.listdir(path)
    n = 100
    start = time.time()
    for i in range(n):
        for file in files:
            tensor = torch.load(path + file)


    end = time.time()
    print(f"Time taken to load {n} files: {end - start}")

def test_numpy():
    files = os.listdir("/extdata4/baeklab/Hyeonseo/m6A/dataset/ver082724/score-perfect/train/pos/")
    n = 100
    start = time.time()
    for i in range(n):
        for file in files:
            data = numpy.load("/extdata4/baeklab/Hyeonseo/m6A/dataset/ver082724/score-perfect/train/pos/" + file)
            data.close()
    end = time.time()
    print(f"Time taken to load {n} files: {end - start}")


test_torch()
test_numpy()
