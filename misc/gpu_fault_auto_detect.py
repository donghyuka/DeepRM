import torch
import torch.multiprocessing as mp
from tqdm import tqdm

def main_worker(gpu, input1, input2, input3, output_cpu):
    input1 = input1.cuda(gpu).repeat(60, 60)
    input2 = input2.cuda(gpu).repeat(60, 60)
    input3 = input3.cuda(gpu).repeat(60, 60)

    torch.cuda.set_device(gpu)
    with torch.no_grad():
        output_gpu = torch.matmul(torch.matmul(input1, input2), input3)

    output_gpu = output_gpu.cpu()
    output_diff = output_gpu - output_cpu

    diff_mean = output_diff.mean()
    print(f"Rank [{gpu}] diff_mean: {diff_mean}")

    return None


def main_master():
    num_gpu = 8
    dim = 100
    for i in tqdm(range(10000)):
        input1 = torch.rand((dim,dim))
        input2 = torch.rand((dim,dim))
        input3 = torch.rand((dim,dim))
        output = torch.matmul(torch.matmul(input1.repeat(60, 60), input2.repeat(60, 60)), input3.repeat(60, 60))
        mp.spawn(main_worker, nprocs=num_gpu, args=(input1, input2, input3, output), join=True)
    return None


if __name__ == "__main__":
    main_master()

