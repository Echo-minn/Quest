#include "bsk_ops.h"
#include "pytorch_extension_utils.h"
#include "topk/decode_select_k.cuh"

using namespace flashinfer;

namespace {

__global__ void accumulate_counts_kernel(const int32_t* __restrict__ topk_pos, // [y, H, k]
										 int y,
										 int H,
										 int k,
										 int num_pages,
										 int32_t* __restrict__ counts // [H, num_pages]
) {
	int h = blockIdx.x;
	if(h >= H) return;
	int tid = threadIdx.x;
	int stride = blockDim.x;
	int total = y * k;
	const int head_stride = k;
	const int yh_stride = H * head_stride;
	int32_t* counts_row = counts + static_cast<size_t>(h) * num_pages;
	for(int t = tid; t < total; t += stride) {
		int yi = t / k;
		int ki = t % k;
		int32_t pos = topk_pos[static_cast<size_t>(yi) * yh_stride + static_cast<size_t>(h) * head_stride + ki];
		if(0 <= pos && pos < num_pages) {
			atomicAdd(counts_row + pos, 1);
		}
	}
}

} // namespace

// Merge y per-score top-k position lists into one length-j per head by frequency.
void merge_topk_positions(torch::Tensor topk_pos,
						  torch::Tensor pages_indices,
						  torch::Tensor merged_counts_out,
						  torch::Tensor merged_indices_out,
						  torch::Tensor tmp_counts_buf,
						  torch::Tensor select_buf,
						  unsigned int j) {
#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(topk_pos);
	CHECK_INPUT(pages_indices);
	CHECK_INPUT(merged_counts_out);
	CHECK_INPUT(merged_indices_out);
	CHECK_INPUT(tmp_counts_buf);
	CHECK_INPUT(select_buf);
	CHECK_DIM(3, topk_pos);
	CHECK_DIM(2, pages_indices);
	CHECK_DIM(2, merged_counts_out);
	CHECK_DIM(2, merged_indices_out);
	CHECK_DIM(2, tmp_counts_buf);
#endif

	TORCH_CHECK(topk_pos.scalar_type() == torch::kInt32, "topk_pos must be int32");
	TORCH_CHECK(pages_indices.scalar_type() == torch::kInt32, "pages_indices must be int32");
	TORCH_CHECK(tmp_counts_buf.scalar_type() == torch::kInt32, "tmp_counts_buf must be int32");
	TORCH_CHECK(merged_counts_out.scalar_type() == torch::kInt32, "merged_counts_out must be int32");
	TORCH_CHECK(merged_indices_out.scalar_type() == torch::kInt32, "merged_indices_out must be int32");

	auto y = static_cast<int>(topk_pos.size(0));
	auto num_heads = static_cast<int>(topk_pos.size(1));
	auto k = static_cast<int>(topk_pos.size(2));
	auto num_pages = static_cast<int>(pages_indices.size(1));

	TORCH_CHECK(pages_indices.size(0) == num_heads, "pages_indices H mismatch");
	TORCH_CHECK(tmp_counts_buf.size(0) == num_heads && tmp_counts_buf.size(1) == num_pages, "tmp_counts_buf shape mismatch");
	TORCH_CHECK(merged_counts_out.size(0) == num_heads && merged_counts_out.size(1) == static_cast<int64_t>(j), "merged_counts_out shape mismatch");
	TORCH_CHECK(merged_indices_out.size(0) == num_heads && merged_indices_out.size(1) == static_cast<int64_t>(j), "merged_indices_out shape mismatch");

	// Current kernels assume 32 heads for best specialization (Llama-7B)
	TORCH_CHECK(num_heads == 32, "merge_topk_positions currently expects num_heads == 32");

	// Zero the histogram buffer
	FLASHINFER_CUDA_CALL(cudaMemsetAsync(tmp_counts_buf.data_ptr(), 0, tmp_counts_buf.nbytes()));

	// Accumulate counts per head
	dim3 grid(num_heads);
	dim3 block(256);
	accumulate_counts_kernel<<<grid, block, 0, nullptr>>>(
		reinterpret_cast<const int32_t*>(topk_pos.data_ptr()),
		y,
		num_heads,
		k,
		num_pages,
		reinterpret_cast<int32_t*>(tmp_counts_buf.data_ptr()));

	// Select top-j by frequency, mapping positions -> page ids via pages_indices
	decode_select_k<int32_t, int32_t, 32>(
		reinterpret_cast<const int32_t*>(tmp_counts_buf.data_ptr()),
		reinterpret_cast<const int32_t*>(pages_indices.data_ptr()),
		reinterpret_cast<char*>(select_buf.data_ptr()),
		num_pages,
		static_cast<int32_t>(j),
		reinterpret_cast<int32_t*>(merged_counts_out.data_ptr()),
		reinterpret_cast<int32_t*>(merged_indices_out.data_ptr()),
		true);
}


