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

## Verify Installation

```bash
deeprm --version
deeprm check
```
* If everything is installed correctly, you should see the version of DeepRM and a message indicating that the installation is successful.
* If you encounter CUDA or torch-related errors, make sure you have installed the correct version of PyTorch with CUDA support.

## Build from Source
* DeepRM uses a C++ preprocessing tool for acceleration.
* The C++ preprocessing tool is both provided as a precompiled binary and source code.
* Depending on your system configuration, you may need to build the C++ preprocessing tool from source.
* The C++ source code is located in the `cpp` directory of the DeepRM repository.
* Please refer to the [advanced installtion](advanced-installation.md) page for detailed build instructions.