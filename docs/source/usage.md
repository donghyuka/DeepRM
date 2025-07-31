# 💻 Usage
## Inference usage
![deeprm_inference_pipeline.png](../images/deeprm_inference_pipeline.png)

### Prepare Data
* You can skip this step if your POD5 files are already basecalled to BAM files with move tags.
```bash
dorado basecaller --reference <reference_path> --min-qscore 0 --emit-moves {args.model} {args.pod5} > <raw_bam_path>"
```
* Filter, sort, and index the BAM files:
```bash
samtools view -@ <threads> -bh -F 276 -o <bam_path> <raw_bam_path>
samtools sort -@ <threads> -o <bam_path> <bam_path>
samtools index -@ <threads> <bam_path>
```
* To preprocess the inference data (transcriptome), run the following command:
```bash
deeprm inference prep --input <input_POD5_dir> --output <output_file> --dorado <dorado_dir>
```
* This will create:
    * Inference dataset: /block
    * Filtered mpileup file: /dorado_output.pileup.filtered.pkl
    * Quality check results: /qc/*.png
### Run Inference
* The trained DeepRM model file is attached in the repository: `model/deeprm_model.pt`.
* For inference, run the following command:
```bash
deeprm inference run --model <model_file> --data <data_dir> --output <prediction_dir> --gpu_pool <gpu_pool>
```
* This will create a directory with single-molecule level result files.
* To get a site-level result, run the following command:
```bash
deeprm inference pileup --input <prediction_dir> --output <pileup_dir> --mpileup <filtered_mpileup_file>
```
* This will create a directory with site-level result files.

## Training usage
![deeprm_train_pipeline.png](../images/deeprm_train_pipeline.png)
### Prepare Data
* You can skip this step if your POD5 files are already basecalled to BAM files with move tags.
```bash
dorado basecaller --min-qscore 0 --emit-moves {args.model} {args.pod5} > <bam_path>
samtools index -@ <threads> <bam_path>
```
* To preprocess the training data (synthetic oligonucleotide), run the following command:
```bash
deeprm train prep --input <input_POD5_dir> --output <output_file>
```
* This will create:
    * Training dataset: /block
* To compile the training dataset, run the following command:
```bash
deeprm train compile --input <input_POD5_dir> --output <output_file>
```
* This will create:
    * Training dataset: /block
### Run Training
* To train the model, run the following command:
```bash
deeprm train run --model deeprm_model --data <data_dir> --output <output_dir> --gpu_pool <gpu_pool>
```
* This will create a directory with the trained model file.
