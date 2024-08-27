import time
import os
import argparse
import math
import subprocess
import atexit

## This script is used for data transfer between two Linux servers.
## ssh, pigz, and tar should be installed on both servers.
## pv should be installed on the source server.
## This script is considerably faster than rsync or scp when transferring large number of small files. (Empirically ~3x faster)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type = str, required = True)
    parser.add_argument("--host", "-o", type = str, required = True)
    parser.add_argument("--user", "-u", type = str, required = True)
    parser.add_argument("--hostdir", "-d", type = str, required = True)
    parser.add_argument("--port", "-p", type = int, default = 22)
    parser.add_argument("--comp_level", "-cl", type = int, default = 3)
    parser.add_argument("--local_comp_thr", "-lct", type = int, default = 16)
    parser.add_argument("--remote_comp_thr", "-rct", type = int, default = 8)
    parser.add_argument("--key", "-k", type = str, default = None)
    parser.add_argument("--jump_host", "-j", type = str, default = None)
    parser.add_argument("--jump_user", "-ju", type = str, default = None)
    parser.add_argument("--jump_port", "-jp", type = int, default = 22)
    parser.add_argument("--size", "-sz", type = int, default = None)
    parser.add_argument("--no_size", "-ns", action = "store_true")
    parser.add_argument("--resume", "-r", action = "store_true")
    parser.add_argument("--nocheck", "-nc", action = "store_true")
    args = parser.parse_args()
    return args

def pretty_size(n,pow=0,b=1024,u='B',pre=['']+[p+'i'for p in'KMGTPEZY']):
    ## https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
    pow,n=min(int(math.log(max(n*b**pow,1),b)),len(pre)-1),n*b**pow
    return "%%.%if %%s%%s"%abs(pow%(-pow-1))%(n/b**float(pow),pre[pow],u)


def main():
    args = parse_args()

    print(f"Program start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not args.no_size:
        if args.size is None:
            source_disk_size = os.popen(f'du -sbL {args.source}').read().split("\t")[0]
        else:
            source_disk_size = args.size
        print(f"Size: {pretty_size(int(source_disk_size))}")
    else:
        source_disk_size = "0"

    source_parent = os.path.dirname(args.source)
    source_subdir = os.path.basename(args.source)

    if args.key is None:
        key_option = ""
    else:
        key_option = f"-i {args.key}"

    if args.jump_host is None:
        jump_option = ""
    else:
        if args.jump_user is None:
            args.jump_user = args.user
        if args.jump_port is None:
            args.jump_port = args.port
        jump_option = f"-J {args.jump_user}@{args.jump_host}:{args.jump_port}"

    if args.nocheck:
        ssh_option = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    else:
        ssh_option = ""

    if args.resume:
        ## First, get list of already transferred files from destination.
        ## Then, exclude them from tar command.
        ssh_command = f"ssh {args.user}@{args.host} -p {args.port} {key_option} {jump_option} {ssh_option} 'cd {args.hostdir}/{source_subdir} ; find . ! -type d -printf \"%T@ %Tc %p\\n\" | sort -n '"
        print(f"Command: {ssh_command}")
        out = subprocess.check_output(ssh_command, shell = True).decode("utf-8")
        out = [f"./{source_subdir}" + x.split(" ")[-1][1:] for x in out.split("\n") if len(x) > 0][:-1]

        ## Save files to exclude
        with open("exclude_list.txt", "w") as f:
            f.write("\n".join(out))
        atexit.register(os.system, "rm exclude_list.txt")

        tar_command = f'tar cf - -C {source_parent} ./{source_subdir} --dereference --exclude-from=exclude_list.txt'

        if not args.no_size:
            ## Get total size of already transferred files from destination.
            ## Then, exclude them from pv command.
            ssh_command = f"ssh {args.user}@{args.host} -p {args.port} {key_option} {jump_option} {ssh_option} 'cd {args.hostdir}/{source_subdir} ; du -sbL .'"
            print(f"Command: {ssh_command}")
            out = subprocess.check_output(ssh_command, shell = True).decode("utf-8")
            transferred_size = int(out.split("\t")[0])
            print(f"Transferred size: {pretty_size(transferred_size)}")
            source_disk_size = str(int(source_disk_size) - transferred_size)

    else:
        tar_command = f'tar cf - -C {source_parent} ./{source_subdir} --dereference'

    pv_command = f"pv -s {source_disk_size}"
    pigz_command = f"pigz -{args.comp_level} -p {args.local_comp_thr}"
    ssh_command = f"ssh {args.user}@{args.host} -o ServerAliveInterval=60 -p {args.port} {key_option} {jump_option} 'cd {args.hostdir} ; pigz -dc -p {args.remote_comp_thr} - | tar xf -'"

    composed_command = f"{tar_command} | {pv_command} | {pigz_command} | {ssh_command}"

    print(f"Command: {composed_command}")

    os.system(composed_command)

    if args.resume:
        ## This script can resume interrupted transfer. However, it may result in corrupted files.
        ## Therefore, after resuming, we use rsync checksum to check the identity of transferred files and fix any corrupted files.
        rsync_command = f"rsync -av --delete --checksum --info=progress2 -e 'ssh -p {args.port} {key_option} {jump_option}' {source_parent}/{source_subdir}/ {args.user}@{args.host}:{args.hostdir}/{source_subdir}/"
        print(f"Command: {rsync_command}")
        os.system(rsync_command)

    print(f"Program end: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return None


if __name__ == "__main__":
    main()
