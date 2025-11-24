/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <torch/extension.h>

#include "decode/decode_handler.cuh"
#include "prefill/prefill.cuh"

void apply_rope_in_place(torch::Tensor q,
						 torch::Tensor k,
						 unsigned int past_kv_len,
						 float rope_scale,
						 float rope_theta);

void rms_norm_forward(torch::Tensor input,
					  torch::Tensor weight,
					  torch::Tensor output,
					  float epsilon);

void topk_filtering(torch::Tensor estimated_value,
					torch::Tensor estimated_indices,
					torch::Tensor d_out,
					torch::Tensor indices_out,
					torch::Tensor buf,
					unsigned int page_budget);

// Merge y per-score top-k position lists into one length-j per head by frequency.
// Inputs:
// - topk_pos: [y, num_heads, k] (int32), positions in range [0, num_pages)
// - pages_indices: [num_heads, num_pages] (int32), maps local position -> page id
// - tmp_counts_buf: [num_heads, num_pages] (int32), will be zeroed and used as histogram
// - select_buf: workspace buffer for RAFT select (char tensor on device)
// Output:
// - merged_counts_out: [num_heads, j] (int32), counts of selected pages
// - merged_indices_out: [num_heads, j] (int32), selected page ids (from pages_indices)
// Params:
// - j: desired merged length
void merge_topk_positions(torch::Tensor topk_pos,
						  torch::Tensor pages_indices,
						  torch::Tensor merged_counts_out,
						  torch::Tensor merged_indices_out,
						  torch::Tensor tmp_counts_buf,
						  torch::Tensor select_buf,
						  unsigned int j);

void estimate_attn_score(torch::Tensor q,
						 torch::Tensor o,
						 torch::Tensor metadata_data,
						 torch::Tensor metadata_indices,
						 torch::Tensor metadata_indptr,
						 unsigned int metadata_last_page_len,
						 unsigned int metadata_last_page_idx,
						 unsigned int layout);

void append_kv_cache_prefill(torch::Tensor k,
							 torch::Tensor v,
							 torch::Tensor kv_data,
							 torch::Tensor kv_indices,
							 torch::Tensor kv_indptr,
							 unsigned int kv_last_page_len,
							 unsigned int kv_last_page_idx,
							 torch::Tensor metadata_data,
							 torch::Tensor metadata_indices,
							 torch::Tensor metadata_indptr,
							 unsigned int metadata_last_page_len,
							 unsigned int metadata_last_page_idx,
							 unsigned int layout);

void append_kv_cache_decode(torch::Tensor k,
							torch::Tensor v,
							torch::Tensor kv_data,
							torch::Tensor kv_indices,
							torch::Tensor kv_indptr,
							unsigned int kv_last_page_len,
							unsigned int kv_last_page_idx,
							torch::Tensor metadata_data,
							torch::Tensor metadata_indices,
							torch::Tensor metadata_indptr,
							unsigned int metadata_last_page_len,
							unsigned int metadata_last_page_idx,
							unsigned int layout);

// Lightweight KV-page prefetch (HBM -> caches) for selected physical pages.
// kv_data: (num_layers, capacity, 2, page_size, num_heads, head_dim)
// page_indices: [N] int32, physical page indices in [0, capacity)
void prefetch_kv_pages(torch::Tensor kv_data, torch::Tensor page_indices);

torch::Tensor prefill_with_paged_kv_cache(torch::Tensor q,
										  torch::Tensor kv_data,
										  torch::Tensor kv_indices,
										  unsigned int kv_last_page_len,
										  bool causal,
										  unsigned int layout,
										  bool allow_fp16_qk_reduction,
										  float rope_scale,
										  float rope_theta);

class BatchDecodeWithPagedKVCachePyTorchWrapper {
public:
	static BatchDecodeWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
		return BatchDecodeWithPagedKVCachePyTorchWrapper(layout);
	}
	void BeginForward(torch::Tensor indptr,
					  unsigned int num_qo_heads,
					  unsigned int num_kv_heads,
					  unsigned int head_dim,
					  unsigned int page_size,
					  torch::Tensor empty_data);

	void EndForward();

	void Forward(torch::Tensor q,
				 torch::Tensor o,
				 torch::Tensor paged_kv_data,
				 torch::Tensor paged_kv_indices,
				 torch::Tensor paged_kv_indptr,
				 unsigned int paged_kv_last_page_len,
				 unsigned int paged_kv_last_page_idx,
				 float rope_scale,
				 float rope_theta);

private:
	BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout)
		: kv_layout_(flashinfer::QKVLayout(layout)) { }
	flashinfer::BatchDecodeHandler handler_;
	flashinfer::QKVLayout kv_layout_;
};