/***************************************************************************************************
 *
 * Copyright (C) 2025 Genome4me Incorporated - All Rights Reserved.
 *
 * This software, including its source code, embedded concepts, and associated
 * documentation, is proprietary to Genome4me Incorporated and is protected
 * under trade secret and copyright law. Unauthorized use, copying, modification,
 * distribution, or disclosure to third parties, in whole or in part, is
 * strictly prohibited unless prior written permission is granted by Genome4me
 * Incorporated. Any such unauthorized actions constitute an infringement of
 * the intellectual property rights of Genome4me Incorporated. For licensing
 * inquiries or permissions, please contact Genome4me Incorporated.
 *
 **************************************************************************************************/

#include "ArgumentParser.h"
#include <iostream>
#include <filesystem>
#include <unistd.h>
#include <cstdlib>
#include <getopt.h>
#include <cstring>

#include "build_time.h"

ArgumentParser::ArgumentParser()
{
}

ArgumentParser::~ArgumentParser()
{
}

void ArgumentParser::print_help(const char* program_name)
{
  cout << "Usage: " << program_name << " [OPTIONS]\n"
      << "\nDeepRM Preprocessing - Segment and Normalize Signal\n"
      << "\nRequired arguments:\n"
      << "  -p, --pod5 PATH          POD5 Input directory\n"
      << "  -b, --bam PATH           Dorado BAM file (specifying '-' for stdin)\n"
      << "  -o, --output PATH        Output directory\n"
      << "\nOptional arguments:\n"
      << "  -t, --thread NUM         Number of thread to use (default: " << args.cpu_count << ")\n"
      << "  -q, --qcut NUM           BQ cutoff (default: " << args.qcut << ")\n"
      << "  -k, --chunk NUM          Chunk size (default: " << args.chunk_size << ")\n"
      << "  -z, --max-token-len NUM  Maximum token length (default: " << args.max_token_len << ")\n"
      << "  -s, --sampling NUM       Sampling rate (default: " << args.sampling << ")\n"
      << "  -y, --boi CHAR           Base of interest (default: " << args.base_of_interest << ")\n"
      << "  -e, --kmer-len NUM       k-mer length (default: " << args.kmer_len << ")\n"
      << "  -l, --cb-len NUM         Context block length (default: " << args.cb_len << ")\n"
      << "  -a, --bam-thread NUM     BAM decompression thread per process (default: " <<
        args.bam_threads << ")\n"
      << "  -n, --process-once NUM   Reads per processing batch (default: " << args.process_once <<
        ")\n"
      << "  -f, --dwell-shift NUM    Distance between motor and pore (default: " <<
        args.dwell_shift << ")\n"
      << "  -w, --sig-window NUM     Signal window size (default: " << args.sig_window << ")\n"
      << "  -g, --filter-flag NUM    BAM flag bits to filter (default: " <<
        args.filter_flag << ")\n"
      << "  -Q, --max-queue NUM      Max BAM queue size, 0=auto (default: " <<
        args.max_queue << ")\n"
      << "  -d, --label-div NUM      Label division factor (default: " << args.label_div << ")\n"
      << "  -h, --help               Show this help message\n"
      << "  -v, --version            Show version information\n";
}

void ArgumentParser::print_version()
{
  cout << "DeepRM Preprocessing v1.1.0 " << DEEPRM_BUILD_TIME << "-" << DEEPRM_GIT_HASH << endl;
}

Arguments ArgumentParser::parse(int argc, char* argv[])
{
  int opt;
  bool has_pod5 = false, has_bam = false, has_output = false;

  static struct option long_options[] = {
    {"pod5", required_argument, 0, 'p'},
    {"bam", required_argument, 0, 'b'},
    {"output", required_argument, 0, 'o'},
    {"thread", required_argument, 0, 't'},
    {"qcut", required_argument, 0, 'q'},
    {"chunk", required_argument, 0, 'k'},
    {"max-token-len", required_argument, 0, 'z'},
    {"sampling", required_argument, 0, 's'},
    {"boi", required_argument, 0, 'y'},
    {"kmer-len", required_argument, 0, 'e'},
    {"cb-len", required_argument, 0, 'l'},
    {"bam-thread", required_argument, 0, 'a'},
    {"process-once", required_argument, 0, 'n'},
    {"dwell-shift", required_argument, 0, 'f'},
    {"sig-window", required_argument, 0, 'w'},
    {"filter-flag", required_argument, 0, 'g'},
    {"max-queue", required_argument, 0, 'Q'},
    {"label-div", required_argument, 0, 'd'},
    {"consistency", no_argument, 0, 'C'},
    {"help", no_argument, 0, 'h'},
    {"version", no_argument, 0, 'v'},
    {0, 0, 0, 0}
  };

  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "p:b:o:t:q:k:z:s:y:e:l:a:n:f:w:g:Q:d:Chv", long_options,
                            &option_index)) != -1) {
    switch (opt) {
      case 'p':
        args.pod5_path = optarg;
        has_pod5 = true;
        break;
      case 'b':
        args.bam_path = optarg;
        has_bam = true;
        break;
      case 'o':
        args.output_path = optarg;
        has_output = true;
        break;
      case 't':
        args.cpu_count = atoi(optarg);
        if (args.cpu_count <= 0) {
          cerr << "Error: CPU count must be positive\n";
          exit(1);
        }
        break;
      case 'q':
        args.qcut = atoi(optarg);
        if (args.qcut < 0) {
          cerr << "Error: Quality cutoff must be non-negative\n";
          exit(1);
        }
        break;
      case 'k':
        args.chunk_size = atoi(optarg);
        if (args.chunk_size <= 0) {
          cerr << "Error: Chunk size must be positive\n";
          exit(1);
        }
        break;
      case 'z':
        args.max_token_len = atoi(optarg);
        if (args.max_token_len <= 0) {
          cerr << "Error: Max token length must be positive\n";
          exit(1);
        }
        break;
      case 's':
        args.sampling = atoi(optarg);
        if (args.sampling <= 0) {
          cerr << "Error: Sampling rate must be positive\n";
          exit(1);
        }
        break;
      case 'y':
        args.base_of_interest = optarg[0];
        if (args.base_of_interest != 'A' && args.base_of_interest != 'T' &&
          args.base_of_interest != 'G' && args.base_of_interest != 'C') {
          cerr << "Error: Base of interest must be A, T, G, or C\n";
          exit(1);
        }
        break;
      case 'e':
        args.kmer_len = atoi(optarg);
        if (args.kmer_len <= 0 || args.kmer_len % 2 == 0) {
          cerr << "Error: k-mer length must be positive odd number\n";
          exit(1);
        }
        break;
      case 'l':
        args.cb_len = atoi(optarg);
        if (args.cb_len <= 0 || args.cb_len % 2 == 0) {
          cerr << "Error: Context block length must be positive odd number\n";
          exit(1);
        }
        break;
      case 'a':
        args.bam_threads = atoi(optarg);
        if (args.bam_threads <= 0) {
          cerr << "Error: BAM threads must be positive\n";
          exit(1);
        }
        break;
      case 'n':
        args.process_once = atoi(optarg);
        if (args.process_once <= 0) {
          cerr << "Error: Process once must be positive\n";
          exit(1);
        }
        break;
      case 'f':
        args.dwell_shift = atoi(optarg);
        if (args.dwell_shift < 0) {
          cerr << "Error: Dwell shift must be non-negative\n";
          exit(1);
        }
        break;
      case 'd':
        args.label_div = strtoull(optarg, nullptr, 10);
        if (args.label_div == 0) {
          cerr << "Error: Label division factor must be positive\n";
          exit(1);
        }
        break;
      case 'w':
        args.sig_window = atoi(optarg);
        if (args.sig_window <= 0 || args.sig_window % 2 == 0) {
          cerr << "Error: Signal window size must be positive odd number\n";
          exit(1);
        }
        break;
      case 'g':
        args.filter_flag = atoi(optarg);
        if (args.filter_flag < 0) {
          cerr << "Error: Filter flag must be non-negative\n";
          exit(1);
        }
        break;
      case 'Q':
        args.max_queue = atoi(optarg);
        if (args.max_queue < 0) {
          cerr << "Error: Max queue size must be non-negative\n";
          exit(1);
        }
        break;
      case 'C':
        args.consistency = true;
        break;
      case 'h':
        print_help(argv[0]);
        exit(0);
        break;
      case 'v':
        print_version();
        exit(0);
        break;
      default:
        print_help(argv[0]);
        exit(1);
    }
  }

  if (!has_pod5 || !has_bam || !has_output) {
    cerr << "Error: Required arguments missing\n";
    print_help(argv[0]);
    exit(1);
  }

  // Check if -C option is given with stdin BAM input
  if (args.consistency && args.bam_path == "-") {
    cerr << "Error: Cannot use consistency mode (-C) with stdin BAM input (-b -)\n";
    exit(1);
  }

  return args;
}

void ArgumentParser::validate_arguments()
{
  // Check if POD5 directory exists
  if (!filesystem::exists(args.pod5_path)) {
    cerr << "Error: POD5 directory does not exist: " << args.pod5_path << "\n";
    exit(1);
  }

  if (strcmp(args.bam_path.c_str(), "-") != 0 && !filesystem::exists(args.bam_path)) {
    cerr << "Error: BAM file does not exist: " << args.bam_path << "\n";
    exit(1);
  }

  // Create output directory if it doesn't exist
  filesystem::create_directories(args.output_path);
}
