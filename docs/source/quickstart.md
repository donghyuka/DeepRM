# 🚀 Quickstart
* For demonstration purposes, you can use examples POD5 and BAM files provided in the `examples` directory of the repository.
* You can also use your own POD5 and BAM files.

## RNA Modification Detection
* Estimated time: ~1 hours

1️⃣ **Prepare data**
```bash
deeprm call prep -p inference_example.pod5 -b inference_example.bam -o <prep_dir>
```
* (Alternative) To supply your own POD5 file:
  ```bash
  dorado basecaller --reference <ref_fasta> --min-qscore 0 --emit-moves rna004_130bps_sup@v5.0.0 <pod5_dir> | \
  tee <bam_path> | deeprm call prep -p <pod5_dir> -b - -o <prep_dir>
  ```
    * If Dorado fails due to "illegal memory access", try adding `--chunksize <chunk_size>` option (e.g., chunk_size=12000).

2️⃣ **Run inference**
```bash
deeprm call run -b inference_example.bam -i <prep_dir> -o <pred_dir> -s 1000
```
* Adjust the `-s` (batch size) parameter according to your GPU memory capacity (default: 10000).
* Expected output file:
    *  Site-level detection result file (.bed)
    *  Molecule-level detection result file (.npz)

## Model Training
* Estimated time: ~1 hours

1️⃣ **Prepare unmodified & modified training data**
```bash
deeprm train prep -p training_a_example.pod5 -b training_a_example.bam -o <prep_dir>/a
deeprm train prep -p training_m6a_example.pod5 -b training_m6a_example.bam -o <prep_dir>/m6a
```

2️⃣ **Compile training data**
```bash
deeprm train compile -n <prep_dir>/a/data -p <prep_dir>/m6a/data -o <prep_dir>/compiled
```

3️⃣ **Run training**
```bash
deeprm train run -d <prep_dir>/compiled -o <output_dir> --batch 64
```
* Adjust the `--batch` parameter according to your GPU memory capacity (default: 1024).
* Expected output file:
  *  Trained DeepRM model file (.pt)

