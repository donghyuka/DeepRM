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

#include <vector>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <bits/ostream.tcc>

using namespace std;

struct NormalizationFactors {
  double quantile_a = 0.2;
  double quantile_b = 0.8;
  double shift_mult = 0.48;
  double scale_mult = 0.59;
};

class Utils {
public:
  static double mean_phred(const vector<uint8_t>& phred);
  static vector<float> move_to_dwell(const vector<bool>& move,
                                     double quantile_a, double quantile_b, double shift_mult,
                                     double scale_mult, int sampling = 6);
  static vector<vector<double>> normalize_trim_segment_signal(
    const vector<double>& signal, const vector<bool>& move,
    int sp, int ts, int ns, double quantile_a, double quantile_b,
    double shift_mult, double scale_mult, int sampling = 6);
  static vector<float> segmented_signal_to_block(
    const vector<vector<float>>& signal_segmented,
    const vector<uint16_t>& segment_len_arr,
    int kmer, int sampling, int sig_window, int pad_to);
  static vector<double> segmented_signal_to_block_range(
    const vector<vector<double>>& signal_segmented,
    int start_idx, int end_idx,
    const vector<uint16_t>& segment_len_arr,
    int kmer, int sampling, int sig_window, int pad_to);
  static vector<uint16_t> create_segment_len_arr(
    const vector<vector<float>>& segment_arr, int sampling);
  template <typename T>
  static T quantile(const vector<T>& data, T q);
  static vector<float> concatenate_segments(
    const vector<vector<float>>& segments);
};

// Template implementation
template <typename T>
T Utils::quantile(const vector<T>& data, T q)
{
  if (data.empty()) return 0.0;
  // Need to copy data since we will modify it
  vector<T> sorted_data(data);
  T index = q * (sorted_data.size() - 1);
  int lower = static_cast<int>(index);
  int upper = lower + 1;
  if (upper >= static_cast<int>(sorted_data.size())) {
    // Only need one element - use nth_element for O(n) performance
    nth_element(sorted_data.begin(), sorted_data.begin() + lower, sorted_data.end());
    return sorted_data[lower];
  }
  // Need two elements for interpolation
  // First, partition to find the lower element
  nth_element(sorted_data.begin(), sorted_data.begin() + lower, sorted_data.end());
  T lower_val = sorted_data[lower];

  // Find the upper element - it's the minimum element in the range [lower+1, end)
  auto upper_it = min_element(sorted_data.begin() + upper, sorted_data.end());
  T upper_val = *upper_it;
  T weight = index - lower;

  return (T)(lower_val + weight * (upper_val - lower_val));
}
