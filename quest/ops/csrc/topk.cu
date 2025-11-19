#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

// Note that estimated_indices does not contain the last page
void topk_filtering(torch::Tensor estimated_value,
							 torch::Tensor estimated_indices,
							 torch::Tensor d_out,
							 torch::Tensor indices_out,
							 torch::Tensor buf,
							 unsigned int page_budget) {
	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(estimated_value); // [num_heads, num_pages]
	CHECK_INPUT(estimated_indices); // [num_heads, num_pages]
	CHECK_DIM(2, estimated_value);
	CHECK_DIM(2, estimated_indices);
	#endif

	auto num_heads = estimated_value.size(0);
	auto num_pages = estimated_value.size(1);

	#ifdef BSK_TORCH_CHECK
	CHECK_EQ(num_pages, estimated_indices.size(1));
	CHECK_EQ(num_heads, estimated_indices.size(0));
	CHECK_GE(num_pages, page_budget);
	CHECK_EQ(estimated_indices.scalar_type(), torch::kInt32);
	// CHECK_EQ(32, num_heads); // Not necessary, but for Llama-7b
	CHECK_EQ(page_budget, d_out.size(1));
	CHECK_EQ(page_budget, indices_out.size(1));
	#endif

    // Use torch::topk instead of Raft
    auto topk_result = torch::topk(estimated_value, page_budget, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
    auto topk_vals = std::get<0>(topk_result);
    auto topk_inds = std::get<1>(topk_result); // indices into estimated_value [H, K]

    d_out.copy_(topk_vals);

    // Gather the original indices
    // topk_inds is Long (int64), estimated_indices is Int32.
    // gather requires index to be Long.
    auto gathered_indices = torch::gather(estimated_indices, 1, topk_inds);
    
    indices_out.copy_(gathered_indices);
}