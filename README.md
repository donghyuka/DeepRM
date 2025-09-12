# DeepRM
#### Deep learning for RNA Modification
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-red?logo=github)](https://github.com/vadanamu/DeepRM)
[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![GitHub Repo stars](https://img.shields.io/github/stars/vadanamu/DeepRM?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/vadanamu/DeepRM)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/vadanamu/DeepRM)
![GitHub contributors](https://img.shields.io/github/contributors/vadanamu/DeepRM)
![GitHub language count](https://img.shields.io/github/languages/count/vadanamu/DeepRM)

![deeprm.png](docs/images/deeprm.png)

## Table of Contents
* [✨ Introduction](#-introduction)
* [🎯 Key Features](#-key-features)
* [📦 Installation](#-installation)
* [🚀 Quickstart](#-quickstart)
* [💻 Usage](#-usage)
  * [Inference](#inference-usage)
  * [Training](#training-usage)
* [🔧 Troubleshooting](#-troubleshooting)
* [📐 Architecture](#-architecture)
* [📝 Citation](#-citation)
* [📝 License](#-license)
* [🏛️ Contributors](#-contributors)
* [🏛️ Acknowledgements](#-acknowledgements)


## ✨ Introduction
DeepRM is a deep learning-based framework for RNA modification detection using Nanopore direct RNA sequencing.
This repository contains the source code for training and running DeepRM.

## 🎯 Key Features
* **High accuracy**: Achieves state-of-the-art accuracy in RNA modification detection and stoichiometry measurement.
* **Single-molecule resolution**: Provides single-molecule level predictions for RNA modifications.
* **End-to-end pipeline**: Easy-to-use pipeline from raw reads to site-level predictions.
* **Customizable**: Supports training of custom models.

## 📦 Installation
### Prerequisites
* Linux x86_64
* Python 3.9+
* Pytorch 2.0+
  * https://pytorch.org/get-started/locally/
  * Please ensure that you have installed the correct version of PyTorch with CUDA support if you want to use GPU for inference or training.

#### Optional
* Torchmetrics 0.9.0+ (only for training)
  * ```bash
    python -m pip install torchmetrics
    ```
* Dorado 0.7.3+ (optional, for basecalling)
  * https://github.com/nanoporetech/dorado
* SAMtools 1.16.1+ (optional, for BAM file processing)
  * http://www.htslib.org/

* Python package requirements are listed in `requirements.txt` and will be installed automatically when you install DeepRM.

### Installation options
* Estimated time: ~10 minutes
1. Install via PIP (recommended)
```bash
python -m pip install deeprm
```

2. Install from source (GitHub)

```bash
git clone https://github.com/vadanamu/deeprm
cd deeprm
python -m pip install -U pip
python -m pip install -e .
```
 * If installation fails on old OS (e.g., CentOS 7) due to NumPy, you can try installing older versions of NumPy first:
 * ```bash
    python -m pip install "numpy<2.3.0,>2.0.0"
    python -m pip install -e .
    ```

### Verify Installation

```bash
deeprm --version
deeprm check
```
 * If everything is installed correctly, you should see the version of DeepRM and a message indicating that the installation is successful.
 * If you encounter CUDA or torch-related errors, make sure you have installed the correct version of PyTorch with CUDA support.

### Build from Source
* DeepRM can use a C++-based preprocessing tool for acceleration, which is both provided as a precompiled binary and source code.
* Depending on your system configuration, you may need to build the C++ preprocessing tool from source, located in the `cpp` directory of the DeepRM repository.
* Please refer to the [cpp/README.md](cpp/README.md) page for detailed build instructions.

## 🚀 Quickstart
* For demonstration purposes, you can use examples POD5 and BAM files provided in the `examples` directory of the repository.
* You can also use your own POD5 and BAM files.

### RNA Modification Detection
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

### Model Training
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


## 💻 Usage
### Inference usage
![deeprm_inference_pipeline.png](docs/images/deeprm_inference_pipeline.png)

#### Prepare Data
##### Accelerated preparation (recommended, default)
* This method uses precompiled C++ binary for accelerating the preprocessing step.
```bash
dorado basecaller --reference <ref_fasta> --min-qscore 0 --emit-moves rna004_130bps_sup@v5.0.0 <pod5_dir> | \
tee <bam_path> | deeprm call prep -p <pod5_dir> -b - -o <prep_dir>
```
* If Dorado fails due to "illegal memory access", try adding `--chunksize <chunk_size>` option (e.g., chunk_size=12000).
* If the precompiled binary does not work on your system, please refer to the [cpp/README.md](cpp/README.md) page for detailed build instructions.
* Adjust the `-g (--filter-flag)` parameter according to your needs. If using a genomic reference, you may want to use `-g 260`.

##### Sequential preparation
* This method is slower than the accelerated preparation method, but is supported for cases such as:
    * The POD5 files are already basecalled to BAM files with move tags.
    * You want to run basecalling and preprocessing in separate machines.

* Basecall the POD5 files to BAM files with move tags (skip if already done):
  * If Dorado fails due to "illegal memory access", try adding `--chunksize <chunk_size>` option (e.g., chunk_size=12000).
```bash
dorado basecaller --reference <reference_path> --min-qscore 0 --emit-moves rna004_130bps_sup@v5.0.0 <pod5_dir> > <raw_bam_path>"
```
* Filter, sort, and index the BAM files:
  * Adjust the `-F` parameter according to your needs. If using a genomic reference, you may want to use `-F 260`.
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

#### Run Inference
* The trained DeepRM model file is attached in the repository: `model/deeprm_model.pt`.
* For inference, run the following command:
    * Adjust the `-s` (batch size) parameter according to your GPU memory capacity (default: 10000).
```bash
deeprm call run --model <model_file> --data <data_dir> --output <prediction_dir> --gpu-pool <gpu_pool>
```
* This will create a directory with the site-level and molecule-level result files.
* Optionally, if you used a transcriptomic reference for alignment, you can convert the result to genomic coordinates by supplying a RefFlat/GenePred/RefGene file (`--annot <annotation_file>`).

#### BED file format
* The output BED file contains the following columns:
* ```text
    1. Reference name (chromosome or transcript ID)
    2. Start position (0-based)
    3. End position (start position + 1)
    4. Strand (-1 for reverse, 1 for forward)
    5. DeepRM modification score
    6. DeepRM modification stoichiometry
    7. Number of total reads called as modified or unmodified
    8. Number of reads called as modified
    9. Number of reads called as unmodified
    ```

#### Molecule-level NPZ file format
* The output NPZ file contains the following arrays:
```text
    1. read_id
    2. label_id
    3. pred: modification score (between 0 and 1)
```
* Read ID specification:
    * The UUID4 format read ID (128 bits) is converted to two 64-bit integers for NumPy compatibility.
    * You can convert the two 64-bit integers back to UUID4 using the following Python code:
      ```python
      import numpy as np
      import uuid
      def int_to_uuid(high, low):
          return uuid.UUID(bytes=b"".join([high.tobytes(),low.tobytes()]))
      ```
* Label ID specification:
    * Label ID contains the reference, position, and strand information.
    * You can decode the label ID using the following Python code:
    ```python
    import numpy as np
    def decode_label_id(label_id, label_div = 10**9):
        strand = np.sign(label_id)
        label_id_abs = np.abs(label_id - 1)
        ref_id = label_id_abs // label_div
        pos = label_id_abs % label_div
        return ref_id, pos, strand
    ```
    * Reference ID is extracted from the input BAM file header.
  
### Training usage
![deeprm_train_pipeline.png](docs/images/deeprm_train_pipeline.png)
#### Prepare Data
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
#### Run Training
* To train the model, run the following command:
```bash
deeprm train run --model deeprm_model --data <data_dir> --output <output_dir> --gpu-pool <gpu_pool>
```
* Adjust the `--batch` parameter according to your GPU memory capacity (default: 1024).
* This will create a directory with the trained model file.


## 🔧 Troubleshooting
* If installation fails on old OS (e.g., CentOS 7) due to a NumPy-related error, you can try installing older versions of NumPy first:
  ```bash
  python -m pip install "numpy<2.3.0,>2.0.0"
  python -m pip install -e .
  ```
* If you encounter CUDA or torch-related errors, make sure you have installed the correct version of PyTorch with correct CUDA version support.
* If Dorado fails due to "illegal memory access", try adding `--chunksize <chunk_size>` option (e.g., chunk_size=12000). 
* If DeepRM call fails due to memory error, try reducing the batch size (`-s` option, default: 10000).
* If DeepRM train fails due to memory error, try reducing the batch size (`--batch` option, default: 1024).
* If DeeepRM call preprocess fails due to `libssl.so.1.1` not found error in newer versons of Ubuntu, try  installing `libssl1.1` package:
  * The libssl file can be found at: https://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl
  ```bash
  wget <libssl_file>
  sudo dpkg <libssl_file>
  ```
* If DeepRM call preprocess fails due to memory error, try reducing the number of threads (`-t` option), the preprocessing batch size (`-n` option), or the output chunk size (`-k` option).
* If DeepRM train does not output training-related metrics, try installing `torchmetrics` package:
  ```bash
  python -m pip install torchmetrics
  ```

## 📐 Architecture
![deeprm_architecture.png](docs/images/deeprm_architecture.png)


## 📝 Citation
If you use DeepRM in your research, please cite the following paper:
```{code-block} text
:class: nohighlight
@article{
  title={Comprehensive single-molecule resolution discovery of m6A RNA modification sites in the human transcriptome},
  author={Gihyeon Kang, Hyeonseo Hwang, Hyeonseong Jeon, Heejin Choi, Hee Ryung Chang, Nagyeong Yeo, Junehee Park, Narae Son, Eunkyeong Jeon, Jungmin Lim, Jaeung Yun, Wook Choi, Jae-Yoon Jo, Jong-Seo Kim, Sangho Park, Yoon Ki Kim, Daehyun Baek},
  journal={In review},
  year={In review},
  publisher={In review}
}
```

## 📝 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
<br />DeepRM is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>
by Seoul National University R&DB Foundation and Genome4me Inc.

See the [LICENSE](LICENSE.md) file for details.

## 🏛️ Contributors
This repository is developed and maintained by the following organization:
* **Laboratory of Computational Biology, School of Biological Sciences, Seoul National University**
    * Principal Investigator: Prof. Daehyun Baek
* **Genome4me, Inc., Seoul, Republic of Korea**


## 🏛️ Acknowledgements
This study was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT, Republic of Korea (MSIT) (RS-2019-NR037866, RS-2020-NR049252, RS-2020-NR049538, and RS-2022-NR067483), by a grant of Korean ARPA-H Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (RS-2025-25422732), by Artificial Intelligence Industrial Convergence Cluster Development Project funded by MSIT and Gwangju Metropolitan City, by National IT Industry Promotion Agency (NIPA) funded by MSIT, and by Korea Research Environment Open Network (KREONET) managed and operated by Korea Institute of Science and Technology Information (KISTI).
