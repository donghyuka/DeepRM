import time
import os
import argparse


## This script is used for data transfer between two servers.
## ssh, pigz, and tar should be installed on both servers.
## pv should be installed on the source server.
## This script is considerably faster than rsync or scp.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type = str, required = True)
    parser.add_argument("--ip", "-i", type = str, required = True)
    parser.add_argument("--user", "-u", type = str, required = True)
    parser.add_argument("--dir", "-d", type = str, required = True)
    parser.add_argument("--port", "-p", type = int, default = 22)
    parser.add_argument("--comp_level", "-cl", type = int, default = 3)
    parser.add_argument("--local_comp_thr", "-lct", type = int, default = 8)
    parser.add_argument("--remote_comp_thr", "-rct", type = int, default = 8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source_dir = args.source
    dest_ip = args.ip
    dest_user = args.user
    dest_dir = args.dir
    dest_port = args.port

    print(f"Program start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    source_disk_size = os.popen(f'du -sh {source_dir}').read().split("\t")[0]
    print(f"Size: {source_disk_size}")

    source_parent = os.path.dirname(source_dir)
    source_subdir = os.path.basename(source_dir)

    tar_command = f'tar cf - -C "{source_parent}" "./{source_subdir}"'
    pv_command = f"pv -s {source_disk_size}"
    pigz_command = f"pigz -{args.comp_level} -p {args.local_comp_thr}"
    ssh_command = f"ssh {dest_user}@{dest_ip} -p {dest_port}  'cd {dest_dir} ; pigz -dc -p {args.remote_comp_thr} - | tar xf -'"

    composed_command = f"{tar_command} | {pv_command} | {pigz_command} | {ssh_command}"

    print(f"Command: {composed_command}")

    os.system(composed_command)

    print(f"Program end: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return None


if __name__ == "__main__":
    main()
