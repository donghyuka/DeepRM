host=$1 #"ubuntu@xxx.xxx.xxx.xxx"
key=$2 #"/extdata4/baeklab/Hyeonseo/m6A/xxx.pem"
version=$3 #"verxxxxxx"

size=$(du -sb /extdata4/baeklab/Hyeonseo/m6A/dataset/"$version" | awk '{print $1}')
tar cf - -C /extdata4/baeklab/Hyeonseo/m6A/dataset ./"$version" | pv -s "$size" | pigz -3 -p 20 | ssh "$host" -i "$key" "cd /data/Hyeonseo/m6A/dataset ; pigz -dc -p 4 - | tar xf - "