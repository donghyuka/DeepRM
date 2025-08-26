# 💻 Usage
## Inference usage
![deeprm_inference_pipeline.png](../images/deeprm_inference_pipeline.png)

### Prepare Data
#### Accelerated preparation (recommended, default)
```bash
dorado basecaller --reference <ref_fasta> --min-qscore 0 --emit-moves rna004_130bps_sup@v5.0.0 <pod5_dir> | \
tee >(deeprm call prep -p <pod5_dir> -b - -o <prep_dir>) >(<bam_path>)
```

#### Sequential preparation
* This method is slower than the accelerated preparation method, but is supported for cases such as:
  * The POD5 files are already basecalled to BAM files with move tags.
  * You want to run basecalling and preprocessing in separate machines.

* Basecall the POD5 files to BAM files with move tags (skip if already done):
```bash
dorado basecaller --reference <reference_path> --min-qscore 0 --emit-moves rna004_130bps_sup@v5.0.0 <pod5_dir> > <raw_bam_path>"
```
* Filter, sort, and index the BAM files:
```bash
samtools view -@ <threads> -bh -F 276 -o <bam_path> <raw_bam_path>
samtools sort -@ <threads> -o <bam_path> <bam_path>
samtools index -@ <threads> <bam_path>
```
* To preprocess the inference data (transcriptome), run the following command:
```bash
deeprm call prep --input <input_POD5_dir> --output <output_file> --dorado <dorado_dir>
```
* This will create the npz files for inference.

### Run Inference
* The trained DeepRM model file is attached in the repository: `model/deeprm_model.pt`.
* For inference, run the following command:
  * Modify the '-bs' (batch size) parameter according to your GPU memory capacity (default: 1000).
```bash
deeprm call run --model <model_file> --data <data_dir> --output <prediction_dir> --gpu_pool <gpu_pool>
```
* This will create a directory with the result files.

## Training usage
![deeprm_train_pipeline.png](../images/deeprm_train_pipeline.png)
### Prepare Data
* You can skip this step if your POD5 files are already basecalled to BAM files with move tags.
```bash
dorado basecaller --min-qscore 0 --emit-moves rna004_130bps_sup@v5.0.0 <pod5_dir> > <bam_path>
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
