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

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <vector>
#include <htslib/sam.h>
#include "../args/ArgumentParser.h"
#include "../pod5/Pod5Reader.h"
#include "../npz/NpzWriter.h"
#include "../merger/RecordMerger.h"
#include "../bam/BamReader.h"

using namespace std;

namespace deeprm {
  class MergedDataWorker {
  private:
    int worker_id;
    const Arguments& args;
    const vector<vector<Pod5RecordMeta>>& pod5_file_meta_records;
    sam_hdr_t* bam_header;

    queue<bam1_t*> bam_queue;
    mutex queue_mutex;
    condition_variable queue_cv;
    atomic<bool> is_running;
    thread worker_thread;

    NpzWriter* writer;
    int output_index;
    int current_file_index;
    vector<ProcessedRecord> pending_records;

    void process_loop();
    bool convert_bam1_to_record(bam1_t* read, sam_hdr_t* header, BamRecord& record) const;
    static vector<pair<int32_t, int32_t>> get_aligned_pairs(bam1_t* read,
                                                            char boi, sam_hdr_t* header);
    static string build_alignment_sequence(bam1_t* read);
    static string build_reference_sequence(bam1_t* read);
    static uint32_t get_md_reference_length(const char* md_tag);

  public:
    MergedDataWorker(int id, const Arguments& args,
                     const vector<vector<Pod5RecordMeta>>& pod5_meta);
    ~MergedDataWorker();

    void start();
    void stop();
    void signal_stop();
    void wait_for_completion();
    void add_bam_data(bam1_t* read);
    void set_bam_header(sam_hdr_t* header) { bam_header = header; }
  };
}
