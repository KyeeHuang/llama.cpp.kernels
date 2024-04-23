#define VDR_Q8_0_Q8_1_MMVQ 2

static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, float * dst,
                  const int ncols, const int nrows, cudaStream_t stream)
{
  GGML_ASSERT(ncols % QK8_0 == 0);
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(1, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  // QK8_0 = 32, QI8_0 = 8
  mul_mat_vec_q<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
    <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

template<int vdr>
static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int *v, const int *u, const float & d8_0, const float & d8_1) 
{
#if __CUDA_ARCH__ >= MIN_CC_DP4A
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    sumi = __dp4a(v[i], u[i], sumi);
  }
  return d8_0 * d8_1 * sumi;
#else
  assert(false);
  return 0.0f;
#endif
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
  const void* __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) 
{
  const block_q8_0* bq8_0 = (const block_q8_0 *) vbq;
  
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs+i);
  }

  return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, bq8_1->ds.x);
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy,
      float* __restrict__ dst, const int ncols, const int nrows)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row >= nrows) return;
  
  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  float tmp = 0.0f;
  
  const block_q_t * x = (const block_q_t * ) vx;
  const block_q8_1 * y = (const block_q8_1 *) vy;
  
  for (int i = 0; i < blocks_per_row; i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i + threadIdx.x / (qi / vdr);

    const int iby = (i + threadIdx.x / (qi / vdr)) * (qk / QK8_1);
    const int iqs = vdr * (threadIdx.x % (qi / vdr));

    tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
  } 
  
  if (threadIdx.x == 0)
    dst[row] = tmp;
}

