import os, argparse, tqdm, gc, glob
from utils.utils import printmessage


def restructure_directory(input_path, dir_keyword = "xxx", rename = "ON7777"):
    ## TODO: This function is not working properly. It should be fixed.
    ## Check if pod5 files exist
    pod5_list = os.listdir(input_path)
    pod5_list = [i for i in pod5_list if i.endswith(".pod5")]

    if len(pod5_list) == 0:
        printmessage(f"No pod5 files found in {input_path}. Attempting directory restructuring")

        if rename == "":
            raise ValueError(f"No pod5 files found in {input_path}. Please specify --rename argument")

        else:
            printmessage(f"No pod5 files found in {input_path}. Attempting directory restructuring")

        rename_path = os.path.join(os.path.dirname(input_path), rename)
        os.rename(input_path, rename_path)
        input_path = rename_path

        ## check if input directory has a single subdirectory
        subdir_list = os.listdir(input_path)
        if len(subdir_list) != 1:
            raise ValueError(f"Input directory {input_path} contains multiple subdirectories.")

        subdir_path = os.path.join(input_path, subdir_list[0])
        ## check if input directory has a single subdirectory
        subdir_list = os.listdir(subdir_path)
        if len(subdir_list) != 1:
            raise ValueError(f"Input directory {subdir_path} contains multiple subdirectories.")

        subsubdir_path = os.path.join(subdir_path, subdir_list[0])
        rename_path = os.path.join(input_path, dir_keyword)
        os.rename(subsubdir_path, rename_path)
        os.rmdir(subdir_path)
        input_path = rename_path

        ## check if input directory has "pod5_pass", "pod5_fail" subdirectories
        subdir_list = os.listdir(input_path)
        if "pod5_pass" not in subdir_list or "pod5_fail" not in subdir_list:
            raise ValueError(f"Input directory {input_path} does not contain pod5_pass and pod5_fail subdirectories.")
        os.makedirs(f"{input_path}/pod5", exist_ok=True)

        ## move pod5 files to pod5 directory
        for subdir in ["pod5_pass", "pod5_fail"]:
            subdir_path = os.path.join(input_path, subdir)
            cmd = f"mv {subdir_path}/*.pod5 {input_path}/pod5/"
            os.system(cmd)

        ## check if empty directories remain
        for subdir in ["pod5_pass", "pod5_fail"]:
            subdir_path = os.path.join(input_path, subdir)
            if len(os.listdir(subdir_path)) == 0:
                os.rmdir(subdir_path)

        ## set input directory to pod5 directory
        input_path = f"{input_path}/pod5"
        printmessage(f"Input directory restructured to {input_path}")

    else:
        pass

    return None

restructure_directory("/extdata4/baeklab/Hyeonseo/m6A/runs/test/ON9999")