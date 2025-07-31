# 📦 Installation

## Prerequisites
* Linux x86_64
* Python 3.9+
* Pytorch 2.0+ (with CUDA support for GPU inference)
    * https://pytorch.org/get-started/locally/
* Torchmetrics 0.9.0+ (for training)
    * ```bash
      python -m pip install torchmetrics
      ```

### Optional
* Dorado 0.7.3+ (optional, for basecalling)
    * https://github.com/nanoporetech/dorado
* SAMtools 1.16.1+ (optional, for BAM file processing)
    * http://www.htslib.org/

* Python package requirements are listed in `requirements.txt` and will be installed automatically when you install DeepRM.

## Installation options
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

## Verify Installation

```bash
deeprm --version
deeprm check
```
* If everything is installed correctly, you should see the version of DeepRM and a message indicating that the installation is successful.
* If you encounter CUDA or torch-related errors, make sure you have installed the correct version of PyTorch with CUDA support.
