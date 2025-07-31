# 🚀 Quickstart
* For demonstration purposes, DeepRM will automatically use examples POD5 and BAM files provided in the repository.
* You can also use your own POD5 and BAM files.

## Inference
```bash
# Prepare data
deeprm inference prep -p inference_example.pod5 -b inference_example.bam -o <prep_dir>
# Run inference
deeprm inference run -d <prep_dir> -o <pred_dir>
# Generate site-level results
deeprm inference pileup -i <pred_dir> -o <pileup_dir> --bed
```

## Training
```bash
# Prepare unmodified data
deeprm train prep -p training_a_example.pod5 -b training_a_example.bam -o <prep_dir>/a
 # Prepare modified data
deeprm train prep -p training_m6a_example.pod5 -b training_m6a_example.bam -o <prep_dir>/m6a
# Compile training data
deeprm train compile -n <prep_dir>/a -p <prep_dir>/m6a -o <prep_dir>/compiled
# Run training
deeprm train run -d <prep_dir>/compiled -o <output_dir> --gpu
```
