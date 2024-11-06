bam_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.bam"
pod5_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/pod5"
ref_path = "/extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta"
wdir = "/extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091"
eventalign_path = f"{wdir}/eventalign.tsv"
dataprep_path = f"{wdir}/dataprep"
output_path = f"{wdir}/out"
model_path = "/extdata4/baeklab/Hyeonseo/m6A/m6anet/rna004.nucleotide.5mer.model.txt "
fastq_path = bam_path[:-4] + ".fastq"
slow5_path = pod5_path[:-4] + "slow5"
blow5_path = pod5_path[:-4] + "merged.blow5"


cmd = f"samtools bam2fq -@ 120 {bam_path} > {fastq_path}"
cmd = f"blue-crab p2s -p 16 -t 8 -K 10000 {pod5_path} -d {slow5_path}"
cmd = f"slow5tools merge -t 120 {slow5_path} -o {blow5_path}"
cmd = f"f5c index -t 120 --slow5 {blow5_path} {fastq_path}"
cmd = f"f5c eventalign --rna -b {bam_path} -r {fastq_path} -g {ref_path} -o {eventalign_path} --kmer-model {model_path} --slow5 {blow5_path} --signal-index --scale-events"
cmd = f"m6anet dataprep --eventalign  {eventalign_path} --out_dir {dataprep_path} --n_processes 120"
cmd = f"m6anet inference --input_dir {dataprep_path} --out_dir {output_path} --pretrained_model HEK293T_RNA004"



cmd = f"samtools bam2fq -@ 120 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.bam \
 > /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.fastq"
cmd = f"blue-crab p2s -p 16 -t 8 -K 10000 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/pod5 \
 -d /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/slow5"
cmd = f"slow5tools merge -t 120 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/slow5 -o /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/merged.blow5"
cmd = (f"f5c index -t 120 --slow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/merged.blow5 \
/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.fastq")
cmd = f"f5c eventalign --rna -b /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.bam \
 -r /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.fastq  \
 -g /extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta  \
 -o /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091/eventalign.tsv \
 --kmer-model /extdata4/baeklab/Hyeonseo/m6A/m6anet/rna004.nucleotide.5mer.model.txt \
 --slow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/merged.blow5 \
 --signal-index --scale-events -x hpc-high"
cmd = f"m6anet dataprep --eventalign /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091/eventalign.tsv \
 --out_dir /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091/dataprep --n_processes 120"
cmd = f"m6anet inference --n_processes 120 --device cuda:all --input_dir /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091/dataprep --out_dir /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091/out --pretrained_model HEK293T_RNA004"


cmd = f"samtools bam2fq -@ 120 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.nosecondary.bam > /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.nosecondary.fastq"
cmd = f"blue-crab p2s -p 16 -t 8 -K 10000 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/raw/pod5 -d /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/raw/slow5"
cmd = f"slow5tools merge -t 120 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/raw/slow5 -o /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/raw/merged.blow5"
cmd = f"f5c index -t 120 --slow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/raw/merged.blow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.nosecondary.fastq"
cmd = f"f5c eventalign --rna -b /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.nosecondary.bam -r /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.nosecondary.fastq -g  /extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta  -o /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/eventalign.tsv --kmer-model /extdata4/baeklab/Hyeonseo/m6A/m6anet/rna004.nucleotide.5mer.model.txt --slow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/raw/merged.blow5 --signal-index --scale-events"
cmd = f"m6anet dataprep --eventalign /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/eventalign.tsv --out_dir  /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/dataprep --n_processes 120"
cmd = f"m6anet inference --input_dir  /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/dataprep --out_dir  /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/out --pretrained_model HEK293T_RNA004"
