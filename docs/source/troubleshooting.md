# 🔧 Troubleshooting
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

