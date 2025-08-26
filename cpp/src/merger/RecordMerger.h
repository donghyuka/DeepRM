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

#include "../bam/BamReader.h"
#include "../pod5/Pod5Reader.h"

#include "../utils/Utils.h"
#include <vector>
#include <memory>

using namespace std;

namespace deeprm {
  struct ProcessedRecord {
    string read_id;
    vector<uint16_t> segment_len_arr;
    vector<double> signal_token;
    vector<uint8_t> kmer_token;
    vector<float> dwell_motor_token;
    vector<float> dwell_pore_token;
    vector<uint8_t> bq_token;
    int64_t label_id;
  };

  class RecordMerger {
  private:
    NormalizationFactors norm_factors;
    int cb_len;
    int kmer_len;
    int max_token_len;
    int sampling;
    int dwell_shift;
    int sig_window;
    uint64_t label_div;

    vector<BamRecord> bam_records;
    vector<Pod5RecordMeta> pod5_meta_records;

    bool process_merged_record_with_meta(const BamRecord& bam_rec,
                                         const Pod5RecordMeta& pod5_meta,
                                         vector<ProcessedRecord>& output) const;

  public:
    RecordMerger(const NormalizationFactors& nf, int cb_len, int kmer_len,
                 int max_token_len, int sampling, int dwell_shift, int sig_window,
                 uint64_t label_div);

    void add_bam_records(vector<BamRecord>&& records);
    void add_pod5_meta_records(vector<Pod5RecordMeta>&& meta_records);
    vector<ProcessedRecord> merge_and_process_with_meta();
    void clear();
  };
} // namespace deeprm
