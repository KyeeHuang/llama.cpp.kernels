#define WARP_SIZE 32

static void rms_norm_f32_cuda(const float* x, float *dst, const int ncols, 
                              const int nrows, const float eps, cudaStream_t stream) 
{
  GGML_ASSERT(ncols % WARP_SIZE == 0);
  const dim3 block_dims(WARP_SIZE, 1, 1); // (32, 1, 1)

  // one block process one row,  
  rms_norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
}

static __global__ void rms_norm_f32(const float *x, float *dst, 
                                    const int ncols, const float eps)
{
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;
  
  float tmp = 0.0f;
  for (int col = tid; col < ncols; col += WARP_SIZE) {
    const float xi = x[col + row*ncols];
    tmp += xi * xi;
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
  }

  const float mean = tmp / ncols; // mean(x^2)
  const float scale = rsqrtf(mean + eps);
  
  // write back
  for (int col = tid; col < ncols; col += WARP_SIZE) {
    dst[row*ncols + col] = scale * x[row*ncols + col];
  }
}