try:
    import sys
    import time
    import os
    import configparser
    import traceback
except ImportError as e:
    print(f"ImportError: {e}")
    input("Press any key to exit.")
    sys.exit(1)


def install_requirements():
    if os.system("which pv > /dev/null") != 0:
        os.system("sudo apt-get install pv")
    if os.system("which pigz > /dev/null") != 0:
        os.system("sudo apt-get install pigz")
    return None


def windows_to_wsl_path(windows_path):
    wsl_path = windows_path.replace("C:\\", "/mnt/c/").replace("\\", "/")
    if wsl_path.startswith('"') or wsl_path.startswith("'"):
        wsl_path = wsl_path[1:]
    if wsl_path.endswith('"') or wsl_path.endswith("'"):
        wsl_path = wsl_path[:-1]
    if wsl_path.endswith("/"):
        wsl_path = wsl_path[:-1]
    return wsl_path


def main(config_path = "config.toml"):
    print(f"Program start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    install_requirements()

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        config_path = input("Enter the path to the config file: ")

    config_toml = configparser.ConfigParser()
    config_toml.read(config_path)
    source_dir = config_toml["path"]["source"]
    dest_dir = config_toml["path"]["destination"]
    jump_ip = config_toml["ssh"]["jumphost"]
    dest_ip = config_toml["ssh"]["destination"]
    dest_user = config_toml["ssh"]["user"]
    dest_port = config_toml["ssh"]["port"]
    local_comp_thr = config_toml["compression"]["local_thread"]
    remote_comp_thr = config_toml["compression"]["remote_thread"]
    comp_level = config_toml["compression"]["comp_level"]

    source_dir = windows_to_wsl_path(source_dir)
    dest_dir = windows_to_wsl_path(dest_dir)

    run_id = input(f"Enter the Nanopore Run ID: ")
    subdirs = os.listdir(source_dir)
    subdirs = [x for x in subdirs if run_id in x]

    if len(subdirs) == 0:
        print("No matching directories found.")
        source_subdir = input("Enter the name of the source directory manually: ")
    elif len(subdirs) > 1:
        print("Multiple matching directories found.")
        print("Select the source directory from the list below")
        for i, subdir in enumerate(subdirs):
            print(f"{i+1}: {subdir}")
        subdir_idx = input("Enter the index of the source directory: ")
        source_subdir = subdirs[int(subdir_idx)-1]
    else:
        source_subdir = subdirs[0]

    source_subdir = windows_to_wsl_path(source_subdir)

    print(f"Source: {source_dir}/{source_subdir}")

    source_disk_size = os.popen(f'du -sh "{source_dir}/{source_subdir}"').read().split("\t")[0]
    print(f"Size: {source_disk_size}")

    tar_command = f'tar cf - -C "{source_dir}" "./{source_subdir}"'
    pv_command = f"pv -s {source_disk_size}"
    pigz_command = f"pigz -{comp_level} -p {local_comp_thr}"
    ssh_command = f"ssh {dest_user}@{dest_ip} -J {dest_user}@{jump_ip}:{dest_port}  'cd {dest_dir} ; pigz -dc -p {remote_comp_thr} - | tar xf -'"

    composed_command = f"{tar_command} | {pv_command} | {pigz_command} | {ssh_command}"

    os.system(composed_command)

    print(f"Program end: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return None


def error_handler(func):
    try:
        func()
    except:
        print(traceback.format_exc())
    input("Press Enter to exit.")
    sys.exit(1)


if __name__ == "__main__":
    error_handler(main)
