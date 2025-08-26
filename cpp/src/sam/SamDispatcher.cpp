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

#include "SamDispatcher.h"
#include "../merger/MergedDataWorker.h"
#include <iostream>
#include <utility>

namespace deeprm {
  SamDispatcher::SamDispatcher(const Arguments& args)
    : bam_path(args.bam_path), bq_cutoff(args.qcut), base_of_interest(args.base_of_interest),
      is_running(false), workers_ready(false), reading_complete(false),
      bam_file(nullptr), header(nullptr), merged_workers(nullptr), pod5_index(nullptr)
  {
  }

  SamDispatcher::~SamDispatcher()
  {
    stop();
    if (header) {
      sam_hdr_destroy(header);
    }
    if (bam_file) {
      sam_close(bam_file);
    }
  }

  void SamDispatcher::start()
  {
    is_running = true;

    cout << "Starting SAM dispatcher for " << bam_path << endl;
    read_thread = thread(&SamDispatcher::read_loop, this);
  }

  void SamDispatcher::stop()
  {
    // Wait for reading to complete first
    if (read_thread.joinable()) {
      read_thread.join();
    }

    // Now signal workers to stop and wait for dispatch completion
    is_running = false;
    queue_cv.notify_all();

    if (dispatch_thread.joinable()) {
      dispatch_thread.join();
    }

    cout << "SAM dispatcher fully stopped" << endl;
  }

  void SamDispatcher::set_workers(vector<MergedDataWorker*>* workers,
                                  unordered_map<string, int>* index)
  {
    merged_workers = workers;
    pod5_index = index;

    // Set BAM header for all workers
    for (auto* worker : *workers) {
      worker->set_bam_header(header);
    }

    workers_ready = true;

    // Start dispatching queued data to workers
    dispatch_thread = thread(&SamDispatcher::dispatch_loop, this);
  }

  void SamDispatcher::read_loop()
  {
    // Open BAM file (supports stdin with "-")
    bam_file = sam_open(bam_path.c_str(), "r");
    if (!bam_file) {
      cerr << "Failed to open BAM file: " << bam_path << endl;
      is_running = false;
      return;
    }

    header = sam_hdr_read(bam_file);
    if (!header) {
      cerr << "Failed to read BAM header" << endl;
      sam_close(bam_file);
      bam_file = nullptr;
      is_running = false;
      return;
    }

    bam1_t* read = bam_init1();

    while (is_running && sam_read1(bam_file, header, read) >= 0) {
      {
        unique_lock<mutex> lock(queue_mutex);
        internal_queue.push(read);
      }
      queue_cv.notify_one();

      // Allocate new read for next iteration
      read = bam_init1();
    }

    bam_destroy1(read);

    // Mark reading as complete
    reading_complete = true;
    queue_cv.notify_all(); // Notify dispatch_to_workers that reading is done

    cout << "SAM dispatcher finished reading BAM file" << endl;
  }

  void SamDispatcher::dispatch_loop()
  {
    while (!reading_complete || !internal_queue.empty()) {
      unique_lock<mutex> lock(queue_mutex);

      if (internal_queue.empty()) {
        if (reading_complete) break;
        queue_cv.wait(lock);
        continue;
      }

      if (!workers_ready) {
        // Workers not ready yet, wait
        lock.unlock();
        this_thread::sleep_for(chrono::milliseconds(100));
        continue;
      }

      // Get read from queue
      bam1_t* read = internal_queue.front();
      internal_queue.pop();
      lock.unlock();

      // Get read ID
      string read_id;
      // Check pi tag first
      uint8_t* pi_tag = bam_aux_get(read, "pi");
      if (pi_tag) {
        read_id = bam_aux2Z(pi_tag);
      } else {
        // Fall back to query name
        read_id = bam_get_qname(read);
      }

      // Find which worker should process this read
      auto it = pod5_index->find(read_id);
      if (it != pod5_index->end()) {
        int worker_idx = it->second;
        if (worker_idx >= 0 && cmp_less(worker_idx, merged_workers->size())) {
          (*merged_workers)[worker_idx]->add_bam_data(read);
        } else {
          bam_destroy1(read);
        }
      } else {
        // No matching POD5 record, destroy the read
        bam_destroy1(read);
      }
    }

    cout << "SAM dispatcher finished dispatching to workers" << endl;
  }

  string SamDispatcher::get_read_id(bam1_t* read)
  {
    // Check pi tag first
    uint8_t* pi_tag = bam_aux_get(read, "pi");
    if (pi_tag) {
      return {bam_aux2Z(pi_tag)};
    }
    // Fall back to query name
    return {bam_get_qname(read)};
  }
}
