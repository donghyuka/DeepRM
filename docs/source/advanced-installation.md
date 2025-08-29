# 📦 Advanced Installation
## DeepRM Preprocessing (C++)
Preprocesses Oxford Nanopore signal data and DORADO BAM reads prior to DeepRM ML inference

## Features

1. Extract read_id, move tags, and quality information from BAM files
2. Extract signal data and calibration information from POD5 files  
3. Merge the two datasets by read_id
4. Perform signal normalization and segmentation
5. Save results to NPZ files in chunks

## Build Requirements

### Required Libraries

1. **system packages**
   - xz-devel
   - zlib-devel
   - bzip2-devel
   - libcurl-devel
   - libuuid-devel
   - openssl-devel
 
2. **htslib**
   - For reading BAM files
   - Source info
     - gerrit:29418/public/htslib, (branch: 1.22)
   - Automatically cloned and built within the project
   - Build output
     - `./htslib/libhts.a`

3. **cnpy**
   - For generating NPZ files
   - Source info
     - https://github.com/rogersce/cnpy (branch: master)
     - Commit: 4e8810b1a8637695171ed346ce68f6984e585ef4
   - Automatically built within the project
   - Build output
     - `./cnpy/build/libcnpy.a`

4. **pod5-file-format**
   - For reading POD5 files
   - Source info
     - https://github.com/nanoporetech/pod5-file-format/tree/0.3.27
   - Imported as prebuilt binary
     - `./pod5-file-format/libpod5_format.a`
     - `./pod5-file-format/libarrow.a`
     - `./pod5-file-format/libjemalloc_pic.a`
     - `./pod5-file-format/libzstd.a`

### System Requirements

- C++20 compatible compiler (GCC 10+ or Clang 12+)
- autotools (for building htslib)
- Standard development tools (make, pkg-config, etc.)

## Build Instructions

### 1. Build Project

```bash
# Full build (including htslib and cnpy)
make

# Or step-by-step build
make clean      # Clean project files only
make all        # Perform build
```

### 2. Cleanup

```bash
make clean      # Clean project files only
make clean_all  # Clean everything including external libraries
```

## Usage

```bash
Usage: bin/deeprm_preprocess [OPTIONS]

DeepRM Preprocessing - Segment and Normalize Signal

Required arguments:
  -p, --pod5 PATH          POD5 Input directory
  -b, --bam PATH           Dorado BAM file (specifying '-' for stdin)
  -o, --output PATH        Output directory

Optional arguments:
  -t, --thread NUM         Number of thread to use (default: 45)
  -q, --qcut NUM           BQ cutoff (default: 0)
  -k, --chunk NUM          Chunk size (default: 16000)
  -z, --max-token-len NUM  Maximum token length (default: 200)
  -s, --sampling NUM       Sampling rate (default: 6)
  -y, --boi CHAR           Base of interest (default: A)
  -e, --kmer-len NUM       k-mer length (default: 5)
  -l, --cb-len NUM         Context block length (default: 21)
  -a, --bam-thread NUM     BAM decompression thread per process (default: 4)
  -n, --process-once NUM   Reads per processing batch (default: 1000)
  -f, --dwell-shift NUM    Distance between motor and pore (default: 10)
  -w, --sig-window NUM     Signal window size (default: 5)
  -g, --filter-flag NUM    BAM flag bits to filter (default: 276)
  -d, --label-div NUM      Label division factor (default: 1000000000)
  -h, --help               Show this help message
  -v, --version            Show version information
```

## Output File Format

NPZ files follow this naming convention:
- `{worker_id}-{processing_unit_id}-{chunk_id}.npz`
- Last processing unit: `{worker_id}-last-{chunk_id}.npz`
- Last chunk: `{worker_id}-last-last.npz`

Each NPZ file contains the following arrays:
- `segment_len_arr`: Segment length array
- `signal_token`: Signal token
- `kmer_token`: k-mer token
- `dwell_motor_token`: Motor dwell token
- `dwell_pore_token`: Pore dwell token
- `bq_token`: Quality score token
- `label_id`: Label ID
- `read_id`: Read ID

## Project Structure

```
deeprm_preprocess/
├── src/
│   ├── main.cpp              # Main application
│   ├── args/                 # Argument parsing
│   ├── sam/                  # BAM file reading
│   ├── pod5/                 # POD5 file reading
│   ├── merger/               # Record merging and processing
│   ├── npz/                  # NPZ file generation
│   └── utils/                # Utility functions
├── htslib/                   # htslib source code
├── cnpy/                     # cnpy library
├── pod5-file-format/         # POD5 library
├── Makefile                  # Build script
└── README.md                 # This file
```

### Features summary

1. **POD5 File Reading**
   - **Uses actual pod5-file-format C API**
   - Batch-based reading, read_id conversion
   - Signal data and calibration information extraction

2. **BAM File Reading**
   - Extract move tags, quality scores, aligned pairs

3. **Record Merging**
   - read_id-based matching
   - Context block extraction and token generation

4. **Signal Processing**
   - Performed by below functions:
     - `move_to_dwell`
     - `normalize_trim_segment_signal`
     - `segmented_signal_to_block`

5. **NPZ Output**
   - Chunk-based saving
   - All field saving (segment_len_arr, signal_token, etc.)

6. **Multiprocessing**
   - Parallel BAM file parsing
   - Parallel POD5 file processing
   - Independent output per worker

### Performance and Memory

- **Memory Efficiency**: Batch-based processing supports large files
- **Multithreading**: Automatic scaling to match CPU core count
- **Chunk-based Output**: Limited memory usage

### Usage Examples

```bash
# Piped input
CMD_SAM_EMIT_STDOUT | ./deeprm_preprocess -p /path/to/pod5/folder -b - -o output/

# File input
./deeprm_preprocess -p /path/to/pod5/folder -b input.bam -o output/

# Custom settings
./deeprm_preprocess -p /path/to/pod5/folder -b input.bam -o output/ -t 16 -k 16000 -z 250 -q 10
```
