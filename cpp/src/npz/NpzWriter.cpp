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
                           size_t offset, size_t count, const string& filename)
{
  if (count == 0) return;

  // Determine max dimensions in this range
  size_t max_segment_len = 0, max_signal_len = 0, max_kmer_len = 0;
  size_t max_dwell_motor_len = 0, max_dwell_pore_len = 0, max_bq_len = 0;

  for (size_t i = offset; i < offset + count; ++i) {
    const auto& r = records[i];
    max_segment_len = max(max_segment_len, r.segment_len_arr.size());
    max_signal_len = max(max_signal_len, r.signal_token.size());
    max_kmer_len = max(max_kmer_len, r.kmer_token.size());
    max_dwell_motor_len = max(max_dwell_motor_len, r.dwell_motor_token.size());
    max_dwell_pore_len = max(max_dwell_pore_len, r.dwell_pore_token.size());
    max_bq_len = max(max_bq_len, r.bq_token.size());
  }

  // Reuse member flat arrays (capacity preserved across calls)
  flat_segment_len.assign(count * max_segment_len, 0);
  flat_signal.assign(count * max_signal_len, 0.0f);
  flat_kmer.assign(count * max_kmer_len, 0);
  flat_dwell_motor.assign(count * max_dwell_motor_len, 0.0f);
  flat_dwell_pore.assign(count * max_dwell_pore_len, 0.0f);
  flat_bq.assign(count * max_bq_len, 0);
  label_ids.assign(count, 0);
  read_ids.assign(count * 2, 0);

  for (size_t i = 0; i < count; ++i) {
    const auto& r = records[offset + i];
    size_t seg_off = i * max_segment_len;
    size_t sig_off = i * max_signal_len;
    size_t kmer_off = i * max_kmer_len;
    size_t dm_off = i * max_dwell_motor_len;
    size_t dp_off = i * max_dwell_pore_len;
    size_t bq_off = i * max_bq_len;

    copy(r.segment_len_arr.begin(), r.segment_len_arr.end(), flat_segment_len.begin() + seg_off);
    // signal_token is double, flat_signal is float — convert
    for (size_t j = 0; j < r.signal_token.size(); ++j)
      flat_signal[sig_off + j] = static_cast<float>(r.signal_token[j]);
    copy(r.kmer_token.begin(), r.kmer_token.end(), flat_kmer.begin() + kmer_off);
    copy(r.dwell_motor_token.begin(), r.dwell_motor_token.end(), flat_dwell_motor.begin() + dm_off);
    copy(r.dwell_pore_token.begin(), r.dwell_pore_token.end(), flat_dwell_pore.begin() + dp_off);
    copy(r.bq_token.begin(), r.bq_token.end(), flat_bq.begin() + bq_off);
    label_ids[i] = r.label_id;

    uuid_t uuid;
    if (uuid_parse(r.read_id.c_str(), uuid) == 0) {
      int64_t* uuid_as_int64 = reinterpret_cast<int64_t*>(uuid);
      read_ids[i * 2] = uuid_as_int64[0];
      read_ids[i * 2 + 1] = uuid_as_int64[1];
    }
  }

  try {
    cnpy::npz_save(filename, "segment_len_arr", flat_segment_len.data(),
                   {count, max_segment_len}, "w", true);
    cnpy::npz_save(filename, "signal_token", flat_signal.data(),
                   {count, max_signal_len}, "a", true);
    cnpy::npz_save(filename, "kmer_token", flat_kmer.data(),
                   {count, max_kmer_len}, "a", true);
    cnpy::npz_save(filename, "dwell_motor_token", flat_dwell_motor.data(),
                   {count, max_dwell_motor_len}, "a", true);
    cnpy::npz_save(filename, "dwell_pore_token", flat_dwell_pore.data(),
                   {count, max_dwell_pore_len}, "a", true);
    cnpy::npz_save(filename, "bq_token", flat_bq.data(),
                   {count, max_bq_len}, "a", true);
    cnpy::npz_save(filename, "label_id", label_ids.data(), {count}, "a", true);
    cnpy::npz_save(filename, "read_id", read_ids.data(), {count, 2}, "a", true);
  } catch (const exception& e) {
    cerr << "Error saving NPZ file " << filename << ": " << e.what() << endl;
  }
}

void NpzWriter::add_records(vector<ProcessedRecord>&& records)
{
  buffer.insert(buffer.end(), make_move_iterator(records.begin()),
                make_move_iterator(records.end()));

  // Save complete chunks directly from buffer using offset
  size_t offset = 0;
  while (buffer.size() - offset >= static_cast<size_t>(chunk_size)) {
    string filename = generate_filename(false, false);
    save_chunk(buffer, offset, chunk_size, filename);
    chunk_id++;
    offset += chunk_size;
  }

  // Keep only remaining records
  if (offset > 0) {
    buffer.erase(buffer.begin(), buffer.begin() + offset);
  }
}

void NpzWriter::flush()
{
  if (!buffer.empty()) {
    size_t offset = 0;
    while (buffer.size() - offset >= static_cast<size_t>(chunk_size)) {
      string filename = generate_filename(true, false);
      save_chunk(buffer, offset, chunk_size, filename);
      chunk_id++;
      offset += chunk_size;
    }

    // Save remaining records
    if (offset < buffer.size()) {
      string filename = generate_filename(true, true);
      save_chunk(buffer, offset, buffer.size() - offset, filename);
    }
    buffer.clear();
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
