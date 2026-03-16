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
#include <unordered_map>
#include <string>
#include <vector>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include "../args/ArgumentParser.h"

using namespace std;

namespace deeprm {
  class MergedDataWorker; // Forward declaration

  class SamDispatcher {
  private:
    string bam_path;
    int bq_cutoff;
    char base_of_interest;
    int bam_threads;

    queue<bam1_t*> internal_queue;
    mutex queue_mutex;
    condition_variable queue_cv;

    atomic<bool> is_running;
    atomic<bool> workers_ready;
    atomic<bool> reading_complete;
    thread read_thread;
    thread dispatch_thread;

    samFile* bam_file;
    sam_hdr_t* header;

    vector<MergedDataWorker*>* merged_workers;
    unordered_map<string, int>* pod5_index;

    void read_loop();
    void dispatch_loop();

  public:
    SamDispatcher(const Arguments& args);
    ~SamDispatcher();

    void start();
    void stop();
    void set_workers(vector<MergedDataWorker*>* workers, unordered_map<string, int>* index);
    sam_hdr_t* get_header() const { return header; }
  };
}
