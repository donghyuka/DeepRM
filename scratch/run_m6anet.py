# blue-crab p2s -p 16 -t 8 -K 10000 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/pod5 -d /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/slow5
#
# slow5tools merge -t 120 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/slow5 -o /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/merged.blow5
#
# f5c index -t 120 --slow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/merged.blow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.fastq
#
# f5c eventalign --rna -b /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.bam  -r /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/result/dorado-070-aligned/intermediates/dorado_output.nosecondary.fastq   -g /extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta   -o /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091/eventalign.tsv  --kmer-model /extdata4/baeklab/Hyeonseo/m6A/m6anet/rna004.nucleotide.5mer.model.txt  --slow5 /extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0091/ON0091/raw/merged.blow5  --signal-index --scale-events -x hpc-high
#
# m6anet dataprep --eventalign /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/eventalign.tsv --out_dir  /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/dataprep --n_processes 120
#
# m6anet inference --input_dir  /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/dataprep --out_dir  /extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0105/out --pretrained_model HEK293T_RNA004