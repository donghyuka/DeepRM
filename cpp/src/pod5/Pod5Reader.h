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

#pragma once

#include <string>
#include <vector>

// POD5 C API includes
extern "C" {
#include <pod5_format/c_api.h>
}

using namespace std;

namespace deeprm {
  struct Pod5RecordMeta {
    string read_id;
    Pod5FileReader_t* reader; // Direct pointer to POD5 reader
    size_t batch_index;
    size_t row_index;
    size_t sample_count;
  };

  class Pod5Reader {
  private:
    string pod5_path;

    // Helper function to convert binary read_id to string
    string read_id_to_string(const read_id_t& read_id);

  public:
    Pod5Reader(const string& path);
    ~Pod5Reader();

    bool parse_single_pod5_meta(const string& file_path,
                                vector<Pod5RecordMeta>& meta_records);

    // Static method to read pod5 record from metadata
    static bool read_pod5_record(const Pod5RecordMeta& meta,
                                 vector<int16_t>& signal);
  };
} // namespace deeprm
