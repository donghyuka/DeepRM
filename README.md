# DeepRM
#### Deep learning for RNA Modification
![GitHub](https://img.shields.io/github/license/vadanamu/deeprm)
![GitHub Repo stars](https://img.shields.io/github/stars/vadanamu/deeprm?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/vadanamu/deeprm)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/vadanamu/deeprm/CI?label=CI%20Status)
![GitHub issues](https://img.shields.io/github/issues/vadanamu/deeprm)
![GitHub pull requests](https://img.shields.io/github/issues-pr/vadanamu/deeprm)
![GitHub contributors](https://img.shields.io/github/contributors/vadanamu/deeprm)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/vadanamu/deeprm)
![GitHub language count](https://img.shields.io/github/languages/count/vadanamu/deeprm)
![GitHub top language](https://img.shields.io/github/languages/top/vadanamu/deeprm)
![GitHub last commit](https://img.shields.io/github/last-commit/vadanamu/deeprm)

![deeprm.png](docs/images/deeprm.png)

## Table of Contents
* [Introduction](#introduction)
* [Quickstart](#quickstart)
* [Usage](#usage)
  * [Training](#training)
  * [Inference](#inference)
* [Installation](#installation)
  * [Prerequisites](#prerequisites)
  * [Installation options](#installation-options)
* [Architecture](#architecture)
* [License](#license)
* [Citation](#citation)
* [Contributors](#contributors)
* [Acknowledgements](#acknowledgements)


## Introduction
DeepRM is a transformer-based model for RNA modification detection using Nanopore direct RNA sequencing.
This repository contains the source code for training and running DeepRM.


## Quickstart
```bash
# Prepare data
deeprm inference prep -i <raw_dir> -o <prep_dir>
# Run inference
deeprm inference run -m <weights.pt> -d <prep_dir> -o <pred_dir>
# Generate site-level results
deeprm inference pileup -i <pred_dir> -o <pileup_dir>
```

## Installation

### Prerequisites
* Linux
* Python 3.9+
* PyTorch 2.0+ (recommended CUDA 11.8+)
  * https://pytorch.org/get-started/locally/
* Dorado 0.7.3+ (optional, for basecalling)
  * https://github.com/nanoporetech/dorado
* SAMtools 1.16.1+ (optional, for BAM file processing)
  * http://www.htslib.org/

* Python package requirements are listed in `requirements.txt` and will be installed automatically when you install DeepRM.

### Installation options
1. Install via PIP (recommended)

```bash
python -m pip install deeprm
```

2. Install via Conda

```bash
conda install -c conda-forge deeprm
```

3. Install from source (GitHub)

```bash
git clone https://github.com/vadanamu/deeprm
cd deeprm
python -m pip install -U pip
python -m pip install -e .
```

## Usage
### Inference
![deeprm_inference_pipeline.png](docs/images/deeprm_inference_pipeline.png)

#### Prepare Data
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
#### Run Inference
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

### Training
![deeprm_train_pipeline.png](docs/images/deeprm_train_pipeline.png)
#### Prepare Data
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
#### Run Training
* To train the model, run the following command:
```bash
deeprm train run --model deeprm_model --data <data_dir> --output <output_dir> --gpu_pool <gpu_pool>
```
* This will create a directory with the trained model file.


## Architecture
![deeprm_architecture.png](docs/images/deeprm_architecture.png)


## License
see [LICENSE](LICENSE) file for details.


## Citation
If you use DeepRM in your research, please cite the following paper:
```bibtex
@article{
  title={Comprehensive discovery of RNA modification sites in the human transcriptome},
  author={Gihyeon Kang, Hyeonseo Hwang, Hyeonseong Jeon, Heejin Choi, Hee Ryung Chang, Junehee Park, Narae Son, Eunkyeong Jeon, Jungmin Lim, Jaeung Yun, Nagyeong Yeo, Yoon Ki Kim, Daehyun Baek},
  journal={In review},
  year={In review},
  publisher={In review}
}
```

## Contributors
This repositoratory is developed by the following organization:
* **Laboratory of Computational Biology, School of Biological Sciences, Seoul National University**
  * Principal Investigator: Prof. Daehyun Baek
  
This repository is maintained by the following authors:
* **Hyeonseo Hwang**


## Acknowledgements
This work was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT, Republic of Korea (MSIT) (NRF-2019M3E5D3073104, NRF-2020R1A2C3007032, NRF-2020R1A5A1018081, and NRF-2022M3A9I2082294), by Artificial Intelligence Industrial Convergence Cluster Development Project funded by MSIT and Gwangju Metropolitan City, by National IT Industry Promotion Agency (NIPA) funded by MSIT, and by Korea Research Environment Open Network (KREONET) managed and operated by Korea Institute of Science and Technology Information (KISTI).
