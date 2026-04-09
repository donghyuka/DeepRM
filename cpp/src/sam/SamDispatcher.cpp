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

#include <iostream>
#include <utility>

#include "../merger/MergedDataWorker.h"
#include "../utils/Utils.h"

namespace deeprm {
  SamDispatcher::SamDispatcher(const Arguments& args)
    : bam_path(args.bam_path), bq_cutoff(args.qcut), base_of_interest(args.base_of_interest),
      bam_threads(args.bam_threads), process_once(args.process_once),
      max_queue_size(args.max_queue > 0 ? args.max_queue
                                        : args.process_once * 4 * args.cpu_count),
      fixed_queue_size(args.max_queue > 0),
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

    log_info() << "Starting SAM dispatcher for " << bam_path
        << " with " << bam_threads << " decompression threads"
        << ", max_queue=" << max_queue_size << endl;
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

    log_info() << "SAM dispatcher fully stopped" << endl;
  }

  void SamDispatcher::set_workers(vector<MergedDataWorker*>* workers,
                                  unordered_map<string, int>* index)
  {
    merged_workers = workers;
    pod5_index = index;
    if (!fixed_queue_size)
      max_queue_size = process_once * 4 * workers->size();

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
      log_err() << "Failed to open BAM file: " << bam_path << endl;
      is_running = false;
      return;
    }

    // Enable multi-threaded BAM decompression
    if (bam_threads > 1) {
      if (hts_set_threads(bam_file, bam_threads) < 0) {
        log_err() << "Failed to set BAM decompression threads" << endl;
      }
    }

    header = sam_hdr_read(bam_file);
    if (!header) {
      log_err() << "Failed to read BAM header" << endl;
      sam_close(bam_file);
      bam_file = nullptr;
      is_running = false;
      return;
    }

    // Chunk-based BAM reading
    deque<bam1_t*> read_q;

    while (is_running) {
      // Read a chunk into local read_q
      while (read_q.size() < read_chunk_size) {
        bam1_t* read = bam_init1();
        if (sam_read1(bam_file, header, read) < 0) {
          bam_destroy1(read);
          goto read_done;
        }
        read_q.push_back(read);
      }

      // Flush read_q into internal_queue
      {
        unique_lock<mutex> lock(queue_mutex);
        if (internal_queue.size() >= max_queue_size) {
          queue_cv.wait(lock, [this] {
            return internal_queue.size() < max_queue_size || !is_running;
          });
        }
        if (!is_running) break;
        internal_queue.insert(internal_queue.end(), read_q.begin(), read_q.end());
        read_q.clear();
      }
      queue_cv.notify_all();
    }

read_done:
    // Flush remaining reads
    if (!read_q.empty()) {
      unique_lock<mutex> lock(queue_mutex);
      internal_queue.insert(internal_queue.end(), read_q.begin(), read_q.end());
      read_q.clear();
      lock.unlock();
      queue_cv.notify_all();
    }

    reading_complete = true;
    queue_cv.notify_all();
    log_info() << "SAM dispatcher finished reading BAM file" << endl;
  }

  void SamDispatcher::dispatch_loop()
  {
    deque<bam1_t*> dispatch_q;

    while (true) {
      // Pop a chunk from internal_queue
      {
        unique_lock<mutex> lock(queue_mutex);

        if (internal_queue.empty()) {
          if (reading_complete) break;
          queue_cv.wait(lock);
          continue;
        }

        if (!workers_ready) {
          lock.unlock();
          this_thread::sleep_for(chrono::milliseconds(100));
          continue;
        }

        size_t count = min(read_chunk_size, internal_queue.size());
        dispatch_q.insert(dispatch_q.end(),
                          internal_queue.begin(), internal_queue.begin() + count);
        internal_queue.erase(internal_queue.begin(), internal_queue.begin() + count);

        bool should_notify = internal_queue.size() <= max_queue_size / 2;
        lock.unlock();
        if (should_notify) queue_cv.notify_all();
      }

      // Dispatch without holding lock
      for (auto* read : dispatch_q) {
        uint8_t* pi_tag = bam_aux_get(read, "pi");
        string parent_id = pi_tag ? string(bam_aux2Z(pi_tag))
                                  : string(bam_get_qname(read));

        auto it = pod5_index->find(parent_id);
        if (it != pod5_index->end()) {
          int worker_idx = it->second;
          if (worker_idx >= 0 && cmp_less(worker_idx, merged_workers->size()))
            (*merged_workers)[worker_idx]->add_bam_data(read);
          else
            bam_destroy1(read);
        } else {
          bam_destroy1(read);
        }
      }
      dispatch_q.clear();
    }

    log_info() << "SAM dispatcher finished dispatching to workers" << endl;
  }
}
