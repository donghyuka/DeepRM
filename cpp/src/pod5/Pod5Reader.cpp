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

#include "Pod5Reader.h"
#include <iostream>
#include <filesystem>
#include <glob.h>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>

namespace deeprm {
  Pod5Reader::Pod5Reader(const string& path) : pod5_path(path)
  {
  }

  Pod5Reader::~Pod5Reader()
  {
  }

  string Pod5Reader::read_id_to_string(const read_id_t& read_id)
  {
    stringstream ss;
    ss << hex << setfill('0');

    // Convert UUID bytes to standard UUID string format
    for (int i = 0; i < 4; ++i) ss << setw(2) << static_cast<unsigned>(read_id[i]);
    ss << "-";
    for (int i = 4; i < 6; ++i) ss << setw(2) << static_cast<unsigned>(read_id[i]);
    ss << "-";
    for (int i = 6; i < 8; ++i) ss << setw(2) << static_cast<unsigned>(read_id[i]);
    ss << "-";
    for (int i = 8; i < 10; ++i) ss << setw(2) << static_cast<unsigned>(read_id[i]);
    ss << "-";
    for (int i = 10; i < 16; ++i) ss << setw(2) << static_cast<unsigned>(read_id[i]);

    return ss.str();
  }

  bool Pod5Reader::parse_single_pod5_meta(const string& file_path,
                                          vector<Pod5RecordMeta>& meta_records)
  {
    try {
      // Check if file exists
      if (!filesystem::exists(file_path)) {
        cerr << "POD5 file does not exist: " << file_path << endl;
        return false;
      }

      // Initialize POD5 library
      pod5_error_t error = pod5_init();
      if (error != POD5_OK) {
        cerr << "Failed to initialize POD5: " << pod5_get_error_string() << endl;
        return false;
      }

      // Open POD5 file
      Pod5FileReader_t* reader = pod5_open_file(file_path.c_str());
      if (!reader) {
        cerr << "Failed to open POD5 file: " << file_path
            << " - " << pod5_get_error_string() << endl;
        pod5_terminate();
        return false;
      }

      // Get batch count
      size_t batch_count = 0;
      error = pod5_get_read_batch_count(&batch_count, reader);
      if (error != POD5_OK) {
        cerr << "Failed to get batch count: " << pod5_get_error_string() << endl;
        pod5_close_and_free_reader(reader);
        pod5_terminate();
        return false;
      }

      // Process each batch
      for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        Pod5ReadRecordBatch_t* batch = nullptr;
        error = pod5_get_read_batch(&batch, reader, batch_index);
        if (error != POD5_OK) {
          cerr << "Failed to get batch " << batch_index
              << ": " << pod5_get_error_string() << endl;
          continue;
        }

        // Get batch row count
        size_t batch_row_count = 0;
        error = pod5_get_read_batch_row_count(&batch_row_count, batch);
        if (error != POD5_OK) {
          cerr << "Failed to get batch row count: " << pod5_get_error_string() <<
              endl;
          pod5_free_read_batch(batch);
          continue;
        }

        // Process each row (read) in batch - metadata only
        for (size_t row = 0; row < batch_row_count; ++row) {
          Pod5RecordMeta meta;

          // Get read information
          ReadBatchRowInfo_t row_info;
          uint16_t read_table_version = 0;
          error = pod5_get_read_batch_row_info_data(
            batch, row, READ_BATCH_ROW_INFO_VERSION, &row_info, &read_table_version);
          if (error != POD5_OK) {
            cerr << "Failed to get row info for row " << row
                << ": " << pod5_get_error_string() << endl;
            continue;
          }

          // Convert read ID to string
          meta.read_id = read_id_to_string(row_info.read_id);

          // Store file path and reader pointer
          meta.file_path = file_path;
          meta.reader = reader;
          meta.batch_index = batch_index;
          meta.row_index = row;

          // Get sample count
          size_t sample_count = 0;
          error = pod5_get_read_complete_sample_count(reader, batch, row, &sample_count);
          if (error != POD5_OK) {
            cerr << "Failed to get sample count for read " << meta.read_id
                << ": " << pod5_get_error_string() << endl;
            continue;
          }
          meta.sample_count = sample_count;

          meta_records.push_back(move(meta));
        }

        // Free batch memory
        pod5_free_read_batch(batch);
      }

      // Note: We do NOT close the reader here as it's stored in metadata
      // The reader should be closed when the metadata is no longer needed

      return true;
    } catch (const exception& e) {
      cerr << "Exception reading POD5 file " << file_path << ": " << e.what() <<
          endl;
      return false;
    }
  }

  bool Pod5Reader::read_pod5_record(const Pod5RecordMeta& meta,
                                    vector<int16_t>& signal)
  {
    try {
      if (!meta.reader) {
        cerr << "Invalid reader pointer in metadata" << endl;
        return false;
      }

      // Get batch
      Pod5ReadRecordBatch_t* batch = nullptr;
      pod5_error_t error = pod5_get_read_batch(&batch, meta.reader, meta.batch_index);
      if (error != POD5_OK) {
        cerr << "Failed to get batch " << meta.batch_index
            << ": " << pod5_get_error_string() << endl;
        return false;
      }

      // Get read information for calibration
      ReadBatchRowInfo_t row_info;
      uint16_t read_table_version = 0;
      error = pod5_get_read_batch_row_info_data(
        batch, meta.row_index, READ_BATCH_ROW_INFO_VERSION, &row_info, &read_table_version);
      if (error != POD5_OK) {
        cerr << "Failed to get row info: " << pod5_get_error_string() << endl;
        pod5_free_read_batch(batch);
        return false;
      }

      // Resize signal vector
      signal.resize(meta.sample_count);

      // Get complete signal data
      error = pod5_get_read_complete_signal(
        meta.reader, batch, meta.row_index, meta.sample_count, signal.data());
      if (error != POD5_OK) {
        cerr << "Failed to get signal for read " << meta.read_id
            << ": " << pod5_get_error_string() << endl;
        pod5_free_read_batch(batch);
        return false;
      }

      // Free batch memory
      pod5_free_read_batch(batch);

      return true;
    } catch (const exception& e) {
      cerr << "Exception reading POD5 record: " << e.what() << endl;
      return false;
    }
  }
} // namespace deeprm
