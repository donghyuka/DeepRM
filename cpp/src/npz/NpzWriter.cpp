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

#include "NpzWriter.h"
#include <filesystem>
#include <iostream>
#include <uuid/uuid.h>

NpzWriter::NpzWriter(const string& output_path, int chunk_size, int worker_id)
  : output_path(output_path), chunk_size(chunk_size), worker_id(worker_id),
    processing_unit_id(1), chunk_id(0)
{
  filesystem::create_directories(output_path);
}

string NpzWriter::generate_filename(bool is_last_processing_unit, bool is_last_chunk)
{
  string filename = output_path + "/" + to_string(worker_id) + "-";

  if (is_last_processing_unit) {
    filename += "last-";
  } else {
    filename += to_string(processing_unit_id) + "-";
  }

  if (is_last_chunk) {
    filename += "last.npz";
  } else {
    filename += to_string(chunk_id) + ".npz";
  }

  return filename;
}

void NpzWriter::save_chunk(const vector<ProcessedRecord>& records,
                           const string& filename)
{
  if (records.empty()) {
    return;
  }

  size_t num_records = records.size();

  // Prepare arrays for each field
  vector<vector<uint16_t>> segment_len_arrays;
  vector<vector<double>> signal_tokens;
  vector<vector<uint8_t>> kmer_tokens;
  vector<vector<float>> dwell_motor_tokens;
  vector<vector<float>> dwell_pore_tokens;
  vector<vector<uint8_t>> bq_tokens;
  vector<int64_t> label_ids;
  vector<int64_t> read_ids;  // UUID as int64 pairs

  // Determine dimensions
  size_t max_segment_len = 0;
  size_t max_signal_len = 0;
  size_t max_kmer_len = 0;
  size_t max_dwell_motor_len = 0;
  size_t max_dwell_pore_len = 0;
  size_t max_bq_len = 0;

  for (const auto& record : records) {
    max_segment_len = max(max_segment_len, record.segment_len_arr.size());
    max_signal_len = max(max_signal_len, record.signal_token.size());
    max_kmer_len = max(max_kmer_len, record.kmer_token.size());
    max_dwell_motor_len = max(max_dwell_motor_len, record.dwell_motor_token.size());
    max_dwell_pore_len = max(max_dwell_pore_len, record.dwell_pore_token.size());
    max_bq_len = max(max_bq_len, record.bq_token.size());
  }

  // Create padded arrays
  for (const auto& record : records) {
    // Segment length array
    vector<uint16_t> padded_segment_len = record.segment_len_arr;
    padded_segment_len.resize(max_segment_len, 0);
    segment_len_arrays.push_back(padded_segment_len);

    // Signal token
    vector<double> padded_signal = record.signal_token;
    padded_signal.resize(max_signal_len, 0.0f);
    signal_tokens.push_back(padded_signal);

    // K-mer token
    vector<uint8_t> padded_kmer = record.kmer_token;
    padded_kmer.resize(max_kmer_len, 0);
    kmer_tokens.push_back(padded_kmer);

    // Dwell motor token
    vector<float> padded_dwell_motor = record.dwell_motor_token;
    padded_dwell_motor.resize(max_dwell_motor_len, 0.0f);
    dwell_motor_tokens.push_back(padded_dwell_motor);

    // Dwell pore token
    vector<float> padded_dwell_pore = record.dwell_pore_token;
    padded_dwell_pore.resize(max_dwell_pore_len, 0.0f);
    dwell_pore_tokens.push_back(padded_dwell_pore);

    // BQ token
    vector<uint8_t> padded_bq = record.bq_token;
    padded_bq.resize(max_bq_len, 0);
    bq_tokens.push_back(padded_bq);

    // Label ID
    label_ids.push_back(record.label_id);
  }

  // Convert read_id strings (UUID) to int64 pairs
  for (const auto& record : records) {
    uuid_t uuid;
    if (uuid_parse(record.read_id.c_str(), uuid) == 0) {
      // UUID is 16 bytes, we convert to 2 int64
      int64_t* uuid_as_int64 = reinterpret_cast<int64_t*>(uuid);
      read_ids.push_back(uuid_as_int64[0]);
      read_ids.push_back(uuid_as_int64[1]);
    } else {
      // If parsing fails, add zeros
      read_ids.push_back(0);
      read_ids.push_back(0);
    }
  }

  // Flatten arrays for cnpy
  vector<uint16_t> flat_segment_len;
  vector<float> flat_signal;
  vector<uint8_t> flat_kmer;
  vector<float> flat_dwell_motor;
  vector<float> flat_dwell_pore;
  vector<uint8_t> flat_bq;

  for (const auto& arr : segment_len_arrays) {
    flat_segment_len.insert(flat_segment_len.end(), arr.begin(), arr.end());
  }

  for (const auto& arr : signal_tokens) {
    flat_signal.insert(flat_signal.end(), arr.begin(), arr.end());
  }

  for (const auto& arr : kmer_tokens) {
    flat_kmer.insert(flat_kmer.end(), arr.begin(), arr.end());
  }

  for (const auto& arr : dwell_motor_tokens) {
    flat_dwell_motor.insert(flat_dwell_motor.end(), arr.begin(), arr.end());
  }

  for (const auto& arr : dwell_pore_tokens) {
    flat_dwell_pore.insert(flat_dwell_pore.end(), arr.begin(), arr.end());
  }

  for (const auto& arr : bq_tokens) {
    flat_bq.insert(flat_bq.end(), arr.begin(), arr.end());
  }

  // Save to NPZ file
  try {
    cnpy::npz_save(filename, "segment_len_arr", flat_segment_len.data(),
                   {num_records, max_segment_len}, "w", true);
    cnpy::npz_save(filename, "signal_token", flat_signal.data(),
                   {num_records, max_signal_len}, "a", true);
    cnpy::npz_save(filename, "kmer_token", flat_kmer.data(),
                   {num_records, max_kmer_len}, "a", true);
    cnpy::npz_save(filename, "dwell_motor_token", flat_dwell_motor.data(),
                   {num_records, max_dwell_motor_len}, "a", true);
    cnpy::npz_save(filename, "dwell_pore_token", flat_dwell_pore.data(),
                   {num_records, max_dwell_pore_len}, "a", true);
    cnpy::npz_save(filename, "bq_token", flat_bq.data(),
                   {num_records, max_bq_len}, "a", true);
    cnpy::npz_save(filename, "label_id", label_ids.data(),
                   {num_records}, "a", true);
    cnpy::npz_save(filename, "read_id", read_ids.data(),
                   {num_records, 2}, "a", true);
  } catch (const exception& e) {
    cerr << "Error saving NPZ file " << filename << ": " << e.what() << endl;
  }
}

void NpzWriter::add_records(const vector<ProcessedRecord>& records)
{
  buffer.insert(buffer.end(), records.begin(), records.end());

  // Process same as Python code: save when chunk_size is reached
  while (static_cast<int>(buffer.size()) >= chunk_size) {
    vector<ProcessedRecord> chunk(buffer.begin(), buffer.begin() + chunk_size);
    buffer.erase(buffer.begin(), buffer.begin() + chunk_size);

    string filename = generate_filename(false, false);
    save_chunk(chunk, filename);
    chunk_id++;
  }
}

void NpzWriter::flush()
{
  if (!buffer.empty()) {
    // Same logic as Python code: save remaining buffer in chunk units
    while (static_cast<int>(buffer.size()) >= chunk_size) {
      vector<ProcessedRecord> chunk(buffer.begin(), buffer.begin() + chunk_size);
      buffer.erase(buffer.begin(), buffer.begin() + chunk_size);

      string filename = generate_filename(true, false);
      save_chunk(chunk, filename);

      chunk_id++;
    }

    // Save remaining records
    if (!buffer.empty()) {
      string filename = generate_filename(true, true);
      save_chunk(buffer, filename);
      buffer.clear();
    }
  }
}

void NpzWriter::increment_processing_unit()
{
  processing_unit_id++;
  chunk_id = 0;
}

void NpzWriter::reset_chunk_id()
{
  chunk_id = 0;
}
