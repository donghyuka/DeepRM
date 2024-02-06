import glob
path_list = ["/extdata4/baeklab/Hyeonseo/m6A/runs/exp_BB87/ON0093/ON0093/result/dorado/token"]
file_list = [y for x in path_list for y in glob.glob(f"{x}/*.pkl")]
print(file_list)