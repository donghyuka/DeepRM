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

#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>
#include <glob.h>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sys/resource.h>

#include "args/ArgumentParser.h"
#include "bam/BamReader.h"
#include "pod5/Pod5Reader.h"
#include "merger/RecordMerger.h"
#include "merger/MergedDataWorker.h"
#include "sam/SamDispatcher.h"
#include "npz/NpzWriter.h"
#include "utils/Utils.h"

using namespace std;
using namespace deeprm;

vector<string> get_pod5_files(const string& directory)
{
  vector<string> files;
  string pattern = directory + "/*.pod5";
  glob_t glob_result;

  if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0) {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      files.emplace_back(glob_result.gl_pathv[i]);
    }
  }
  sort(files.begin(), files.end());

  globfree(&glob_result);
  return files;
}

void process_pod5_meta_worker(int worker_id, const vector<string>& pod5_files,
                              vector<vector<Pod5RecordMeta>>& pod5_meta_records)
{
  cout << "Starting POD5 metadata worker " << worker_id << endl;

  // Reserve space for expected number of files
  pod5_meta_records.reserve(pod5_files.size());

  // Reuse this vector to avoid repeated allocations
  vector<Pod5RecordMeta> file_meta_records;

  // Process each POD5 file separately to maizntain file-level structure
  int file_index = 0;
  for (const string& pod5_file : pod5_files) {
    file_index++;
    cout << "POD5 metadata worker " << worker_id << " processing " << file_index
        << "/" << pod5_files.size() << " files" << endl;
    Pod5Reader pod5_reader(pod5_file);
    file_meta_records.clear(); // Clear for reuse

    if (!pod5_reader.parse_single_pod5_meta(pod5_file, file_meta_records)) {
      cerr << "Corrupted POD5 file: " << pod5_file << endl;
      continue;
    }

    // Add this file's metadata records as a separate vector
    pod5_meta_records.push_back(move(file_meta_records));
  }

  int total_records = 0;
  for (const auto& file_records : pod5_meta_records) {
    total_records += file_records.size();
  }
}

void process_bam_worker(int worker_id, int num_workers, const Arguments& args,
                        vector<BamRecord>& bam_records,
                        unordered_map<string, int>& ref_index_dict)
{
  cout << "Starting BAM worker " << worker_id << endl;

  BamReader reader(args.bam_path, args.qcut, args.base_of_interest, ref_index_dict);
  auto records = reader.parse_bam(worker_id, num_workers, args.bam_threads);

  size_t record_count = records.size(); // Save size before move
  bam_records = move(records);
  cout << "BAM worker " << worker_id << " processed " << record_count << " records" << endl;
}

void process_merged_data_meta_worker(int pod5_worker_id, const Arguments& args,
                                     const vector<vector<Pod5RecordMeta>>& pod5_file_meta_records,
                                     const unordered_map<string, vector<BamRecord>>& bam_index)
{
  cout << "Starting record merger for pod5 metadata from POD5 worker " << pod5_worker_id << endl;

  // Set normalization factors same as Python code
  NormalizationFactors norm_factors;

  // Create NPZ writer using POD5 worker_id to match Python behavior
  NpzWriter writer(args.output_path, args.chunk_size, pod5_worker_id);

  int output_index = 0;

  // Process each POD5 file separately (same as Python's per-file processing)
  int file_index = 0;
  for (const auto& pod5_meta_records : pod5_file_meta_records) {
    file_index++;
    cout << "Record merger " << pod5_worker_id << " processing " << file_index
        << "/" << pod5_file_meta_records.size() << " batches" << endl;

    if (pod5_meta_records.empty()) {
      continue;
    }

    // Create merged records for this POD5 file (same as Python's pd.merge)

    vector<pair<Pod5RecordMeta, BamRecord>> merged_records;

    for (const auto& pod5_meta : pod5_meta_records) {
      auto bam_it = bam_index.find(pod5_meta.read_id);
      if (bam_it != bam_index.end()) {
        // For each POD5 record, add all BAM records with matching read_id
        for (const auto& bam_rec : bam_it->second) {
          merged_records.emplace_back(pod5_meta, bam_rec);
        }
      }
    }

    if (merged_records.empty()) {
      continue;
    }

    // Split merged records into process_once units (same as Python's np.array_split)
    int num_splits = max(1, static_cast<int>(merged_records.size() / args.process_once));

    // np.array_split divides data as evenly as possible
    size_t base_size = merged_records.size() / num_splits;
    size_t remainder = merged_records.size() % num_splits;

    size_t current_idx = 0;
    for (int split_idx = 0; split_idx < num_splits; split_idx++) {
      // Calculate batch size: first 'remainder' batches get one extra element
      size_t batch_size = base_size + (split_idx < static_cast<int>(remainder) ? 1 : 0);

      if (current_idx >= merged_records.size()) {
        break;
      }

      output_index++;

      // Extract batch records with move semantics
      vector<Pod5RecordMeta> batch_pod5_meta;
      vector<BamRecord> batch_bam;
      batch_pod5_meta.reserve(batch_size);
      batch_bam.reserve(batch_size);

      for (size_t i = 0; i < batch_size && current_idx < merged_records.size(); ++i, ++
           current_idx) {
        batch_pod5_meta.push_back(move(merged_records[current_idx].first));
        batch_bam.push_back(move(merged_records[current_idx].second));
      }

      // Create merger and process
      RecordMerger merger(norm_factors, args.cb_len, args.kmer_len, args.max_token_len,
                          args.sampling, args.dwell_shift, args.sig_window, args.label_div);

      merger.add_bam_records(move(batch_bam));
      merger.add_pod5_meta_records(move(batch_pod5_meta));

      vector<ProcessedRecord> processed_records = merger.merge_and_process_with_meta();

      if (!processed_records.empty()) {
        // Write to NPZ
        writer.add_records(processed_records);
      }

      writer.increment_processing_unit();
    }
  }

  // Flush remaining records (same as Python code)
  writer.flush();
  cout << "Record merger for pod5 metadata from POD5 worker " << pod5_worker_id <<
      " completed" << endl;
}


int main(int argc, char* argv[])
{
  struct rlimit rlim;
  if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
    rlim.rlim_cur = rlim.rlim_max;
    if (setrlimit(RLIMIT_NOFILE, &rlim) == 0) {
      cout << "File descriptor limit set to: " << rlim.rlim_cur << endl;
    } else {
      cerr << "Warning: Failed to set file descriptor limit" << endl;
    }
  }

  auto start_time = chrono::high_resolution_clock::now();
  cout << "Started DeepRM Preprocessing" << endl;

  // Parse arguments
  ArgumentParser parser;
  Arguments args = parser.parse(argc, argv);
  parser.validate_arguments();

  // Get POD5 files
  vector<string> pod5_files = get_pod5_files(args.pod5_path);
  if (pod5_files.empty()) {
    cerr << "No POD5 files found in directory: " << args.pod5_path << endl;
    return 1;
  }

  cout << "Found " << pod5_files.size() << " POD5 files" << endl;

  // Step 1: Start SAM dispatcher and POD5 metadata workers concurrently

  // Start SAM dispatcher
  SamDispatcher sam_dispatcher(args);

  // Check if consistency mode is enabled
  if (!args.consistency) {
    // Normal mode: Use streaming approach with SAM dispatcher
    cout << "Running in normal mode - streaming BAM records" << endl;
    sam_dispatcher.start();
  }

  // Process POD5 files with multiple workers (metadata only)
  vector<thread> pod5_threads;
  vector<vector<vector<Pod5RecordMeta>>> pod5_meta_records(args.cpu_count);

  // Split POD5 files among workers
  int num_pod5_workers = min(static_cast<int>(pod5_files.size()), args.cpu_count);

  cout << "Starting POD5 metadata processing with " << num_pod5_workers << " workers" << endl;

  // Calculate base size and remainder like np.array_split
  size_t base_size = pod5_files.size() / num_pod5_workers;
  size_t remainder = pod5_files.size() % num_pod5_workers;

  size_t current_idx = 0;
  for (int i = 0; i < num_pod5_workers; ++i) {
    // First 'remainder' workers get one extra file
    size_t worker_file_count = base_size + (i < static_cast<int>(remainder) ? 1 : 0);

    if (current_idx < pod5_files.size()) {
      vector<string> worker_files(
        pod5_files.begin() + current_idx, pod5_files.begin() + current_idx + worker_file_count
      );
      pod5_threads.emplace_back(process_pod5_meta_worker, i, worker_files,
                                ref(pod5_meta_records[i]));
      current_idx += worker_file_count;
    }
  }

  // Wait for POD5 workers to complete
  for (auto& t : pod5_threads) {
    t.join();
  }

  // Count POD5 metadata records
  unsigned long count_pod5_records = 0;
  for (const auto& worker_files : pod5_meta_records) {
    for (const auto& file_records : worker_files) {
      count_pod5_records += file_records.size();
    }
  }

  cout << "POD5 metadata processing completed. Total records: " << count_pod5_records << endl;

  // Step 2: Create POD5 index and merged data workers
  cout << "Creating POD5 index for merged data workers" << endl;

  // Create POD5 index (read_id -> worker_id mapping)
  unordered_map<string, int> pod5_index;
  pod5_index.reserve(count_pod5_records);
  for (int worker_id = 0; worker_id < num_pod5_workers; ++worker_id) {
    for (const auto& file_records : pod5_meta_records[worker_id]) {
      for (const auto& record : file_records) {
        pod5_index[record.read_id] = worker_id;
      }
    }
  }

  // Check if consistency mode is enabled
  if (args.consistency) {
    // Consistency mode: Read all BAM records first with multiple workers, and distribute to workers
    cout << "Running in consistency mode - reading all BAM records first" << endl;

    // Parse BAM file with multiple workers
    int num_bam_workers = args.cpu_count / args.bam_threads;
    vector<thread> bam_threads;
    vector<vector<BamRecord>> bam_results(num_bam_workers);

    cout << "Starting BAM parsing with " << num_bam_workers << " workers" << endl;


    samFile* bam_file = sam_open(args.bam_path.c_str(), "r");
    if (!bam_file) {
      cerr << "Failed to open BAM file: " << args.bam_path << endl;
      return 1;
    }

    sam_hdr_t* header = sam_hdr_read(bam_file);
    if (!header) {
      cerr << "Failed to read BAM header" << endl;
      sam_close(bam_file);
      return 1;
    }

    // Build reference index dictionary
    unordered_map<string, int> ref_index_dict;
    for (int i = 0; i < header->n_targets; ++i) {
      ref_index_dict[string(header->target_name[i])] = i;
    }

    sam_hdr_destroy(header);
    sam_close(bam_file);

    for (int i = 0; i < num_bam_workers; ++i) {
      bam_threads.emplace_back(process_bam_worker, i, num_bam_workers, ref(args),
                               ref(bam_results[i]), ref(ref_index_dict));
    }

    // Wait for BAM workers to complete
    for (auto& t : bam_threads) {
      t.join();
    }

    // Merge BAM results
    vector<BamRecord> entire_bam_records;
    // Pre-calculate total size for reserve
    size_t total_bam_records = 0;
    for (const auto& result : bam_results) {
      total_bam_records += result.size();
    }
    entire_bam_records.reserve(total_bam_records);

    for (const auto& result : bam_results) {
      entire_bam_records.insert(entire_bam_records.end(), make_move_iterator(result.begin()),
                                make_move_iterator(result.end()));
    }

    cout << "BAM parsing completed. Total records: " << entire_bam_records.size() << endl;

    // Step 3: Process merged data - one thread per POD5 worker to match Python behavior
    vector<thread> merge_threads;

    cout << "Starting merged data processing with " << num_pod5_workers <<
        " workers (matching POD5 workers)" << endl;

    // Index BAM records by read_id - multiple BAM records can have same read_id
    unordered_map<string, vector<BamRecord>> bam_index;
    for (const auto& bam_rec : entire_bam_records) {
      bam_index[bam_rec.read_id].push_back(bam_rec);
    }

    for (int i = 0; i < num_pod5_workers; ++i) {
      // Each merge worker processes file-by-file metadata from corresponding POD5 worker
      merge_threads.emplace_back(process_merged_data_meta_worker, i, ref(args),
                                 cref(pod5_meta_records[i]), cref(bam_index));
    }

    // Wait for merge workers to complete
    for (auto& t : merge_threads) {
      t.join();
    }

    // Stop SAM dispatcher (not used in consistency mode, but need to clean up)
    sam_dispatcher.stop();
  } else {
    // Create merged data workers
    vector<MergedDataWorker*> merged_workers;
    for (int i = 0; i < num_pod5_workers; ++i) {
      auto worker = new MergedDataWorker(i, args, pod5_meta_records[i]);
      worker->start();
      merged_workers.push_back(worker);
    }

    cout << "Started " << num_pod5_workers << " merged data workers" << endl;

    // Set workers for SAM dispatcher
    sam_dispatcher.set_workers(&merged_workers, &pod5_index);

    // Wait for SAM dispatcher to finish
    sam_dispatcher.stop();

    // Signal all workers to stop (notify them to process final batches)
    for (auto worker : merged_workers) {
      worker->signal_stop();
    }

    // Wait for all workers to finish processing final batches
    for (auto worker : merged_workers) {
      worker->wait_for_completion();
      delete worker;
    }
  }

  auto end_time = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
  double runtime_minutes = duration.count() / 60000.0;

  cout << "Finished DeepRM Preprocessing" << endl;
  cout << fixed << setprecision(2) << "Total runtime: " << runtime_minutes << " minutes" << endl;
  return 0;
}
