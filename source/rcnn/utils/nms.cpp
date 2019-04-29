#include "nms.h"
#include <torch/torch.h>
// #include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
// #include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>


int const threadsPerBlock = sizeof(unsigned long long) * 8;

// __device__ inline float devIoU(float const * const a, float const * const b) {
//   float left = max(a[0], b[0]), right = min(a[2], b[2]);
//   float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
//   float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
//   float interS = width * height;
//   float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
//   float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
//   return interS / (Sa + Sb - interS);
// }

// __global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
//                            const float *dev_boxes, unsigned long long *dev_mask) {
//   const int row_start = blockIdx.y;
//   const int col_start = blockIdx.x;

//   // if (row_start > col_start) return;

//   const int row_size =
//         min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
//   const int col_size =
//         min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

//   __shared__ float block_boxes[threadsPerBlock * 5];
//   if (threadIdx.x < col_size) {
//     block_boxes[threadIdx.x * 5 + 0] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
//     block_boxes[threadIdx.x * 5 + 1] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
//     block_boxes[threadIdx.x * 5 + 2] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
//     block_boxes[threadIdx.x * 5 + 3] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
//     block_boxes[threadIdx.x * 5 + 4] =
//         dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
//   }
//   __syncthreads();

//   if (threadIdx.x < row_size) {
//     const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
//     const float *cur_box = dev_boxes + cur_box_idx * 5;
//     int i = 0;
//     unsigned long long t = 0;
//     int start = 0;
//     if (row_start == col_start) {
//       start = threadIdx.x + 1;
//     }
//     for (i = start; i < col_size; i++) {
//       if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
//         t |= 1ULL << i;
//       }
//     }
//     const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
//     dev_mask[cur_box_idx * col_blocks + col_start] = t;
//   }
// }

// // boxes is a N x 5 tensor
// torch::Tensor nms_cuda(const torch::Tensor boxes, float nms_overlap_thresh) {
//   using scalar_t = float;
//   AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
//   auto scores = boxes.select(1, 4);
//   auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
//   auto boxes_sorted = boxes.index_select(0, order_t);

//   int boxes_num = boxes.size(0);

//   const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

//   scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

//   THCState *state = torch::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

//   unsigned long long* mask_dev = NULL;
//   //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
//   //                      boxes_num * col_blocks * sizeof(unsigned long long)));

//   mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

//   dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
//               THCCeilDiv(boxes_num, threadsPerBlock));
//   dim3 threads(threadsPerBlock);
//   nms_kernel<<<blocks, threads>>>(boxes_num,
//                                   nms_overlap_thresh,
//                                   boxes_dev,
//                                   mask_dev);

//   std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
//   THCudaCheck(cudaMemcpy(&mask_host[0],
//                         mask_dev,
//                         sizeof(unsigned long long) * boxes_num * col_blocks,
//                         cudaMemcpyDeviceToHost));

//   std::vector<unsigned long long> remv(col_blocks);
//   memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

//   torch::Tensor keep = torch::empty({boxes_num}, boxes.options().dtype(torch::kLong).device(torch::kCPU));
//   int64_t* keep_out = keep.data<int64_t>();

//   int num_to_keep = 0;
//   for (int i = 0; i < boxes_num; i++) {
//     int nblock = i / threadsPerBlock;
//     int inblock = i % threadsPerBlock;

//     if (!(remv[nblock] & (1ULL << inblock))) {
//       keep_out[num_to_keep++] = i;
//       unsigned long long *p = &mask_host[0] + i * col_blocks;
//       for (int j = nblock; j < col_blocks; j++) {
//         remv[j] |= p[j];
//       }
//     }
//   }

//   THCudaFree(state, mask_dev);
//   // TODO improve this part
//   return std::get<0>(order_t.index({keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)}).sort(0, false));
// }


template <typename scalar_t>
torch::Tensor nms_cpu_kernel(const torch::Tensor& dets,
                          const torch::Tensor& scores,
                          const float threshold) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return torch::empty({0}, dets.options().dtype(torch::kLong).device(torch::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  torch::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  torch::Tensor suppressed_t = torch::zeros({ndets}, dets.options().dtype(torch::kByte).device(torch::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return torch::nonzero(suppressed_t == 0).squeeze(1);
}

torch::Tensor nms_cpu(const torch::Tensor& dets,
               const torch::Tensor& scores,
               const float threshold) {
  torch::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}