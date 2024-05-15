// Q * K 
// Q shape [1, 32, 1, 128] K cache shape [1, 32, seq_len, 128] seq_len is tokens number 
// gridDim = {1, seq_len, 32}, blockDim = {32, 1, 1}
static __global__ void mul_mat_p021_f16_f32 (
  const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
  const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) 
{
  const half * x = (const half *) vx;
  
  const int row_x = blockDim.y * blockIdx.y + threadIdx.y;
  const int channel = blockDim.z * blockIdx.z + threadIdx.z;
  const int channel_x = channel / (nchannels_y / nchannels_x);

  const int nrows_y = ncols_x; // 128
  const int nrows_dst = nrows_x; // seq_len
  const int row_dst = row_x; // [0,..,seq_len-1]
  
  float tmp = 0.0f;
  
  for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
    const int col_x = col_x0 + threadIdx.x;
  
    if (col_x >= ncols_x) break;
    
    // x is transposed and permuted
    // calc. k cache index
    // k cache in memory [seq_len, multihead, head_dim]
    // index calc. method:
    const int ix = row_x * nchannels_x * ncols_x + channel_x * ncols_x + col_x;
    // fp16->fp32 to save memory
    const float xi = __half2float(x[ix]);
    // k cache's row index equal to Q's row index
    const int row_y = col_x;
  
    // y is not transposed but premuted
    const int iy = channel*nrows_y + row_y; // calc. Q's global index
    tmp += xi * y[iy];
  }

  // dst is not transposed and not premuted
  // dst's shape = [32, 1, seq_len], memory alloc is [32, seq_len]
  // dst index:
  const int idst = channel*nrows_dst + row_dst;

  // sum up partial sums and write back result
  // 又是熟悉的 block's sum
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
  }

  if (threadIdx.x == 0) dst[idst] = tmp;
}

static __global__ void scale_f32(const float* x, float * dst, 
                                 const float scale, const int k)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (i >= k) return;

  dst[i] = scale * x[i]; 
}

// attention mask
static __global__ void diag_mask_inf_f32 (const float* x, float * dst,
                  const int ncols, const int rows_per_channel, const int n_past)
{
  const int col = blockDim.x * blockIdx.x + threadIdx.y;
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (col >= ncols) return;
  
  const int i = row * ncols + col;
  // dst[i] = col > n_past + row ? -INFINITY : x[i];
  dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX;
  // equivalent within rounding error but slightly faster on GPU
  
} 

// Attention Score * V 
// Attention Score's shape [1, 32, 1, seq_len] V's shape [1, 32, seq_len, 128]
// gridDim = {1, 128, 32}, blockDim = {32, 1, 1}
static __global__ void mul_mat_vec_nc_f16_f32 (
  const void * __restrict__ vx, const float * __restrict__ y, float* __restrict__ dst,
  const int ncols_x, const int nrows_x, const int row_stride_x, const int channel_stride_x,
  const int channel_x_divisor) {
  
  const half * x = (const half *) vx; // V cache 存储用的 fp16
  
  const int row_x = blockDim.y * blockIdx.y + threadIdx.y; // index of head_dim -> 0-127
  const int channel = blockDim.z * blockIdx.z + threadIdx.z; // index of multi-head -> 0-31
  const int channel_x = channel / channel_x_divisor; // channel/1

  const int nrows_y = ncols_x; // seq_len
  const int nrows_dst = nrows_x; // 128
  const int row_dst = row_x; // index of head_dim -> 0-127
  
  // Attention Score * V's shape [1,32,1,128]
  // dst = (index of multi-head) * 128 + index of head_dim
  const int idst = channel * nrows_dst + row_dst;
  
  float tmp = 0.0f;
  // 循环处理 seq_len 序列，每个线程处理 seq_len/blockDim.x 个数
  for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x ) {
    const int col_x = col_x0 + threadIdx.x;
    
    if (col_x >= ncols_x) break;

    // V cache's index
    const int ix = channel_x * channel_stride_x + row_x * row_stride_x + col_x;
    // fp16 -> fp32
    const float xi = __half2float(x[ix]);
    // Attention Score index
    const int row_y = col_x;
    const int iy = channel * nrows_y + row_y;
    
    tmp += xi * y[iy]; 
  }
  
  // sum up partial sums and write back result
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
  }

  if (threadIdx.x == 0) {
    dst[idst] = tmp;
  }
} 

// add_f32
static __global__ void add_f32 (const float *x, const float *y, float *dst,
                                const int kx, const int ky) 
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i >= kx) {
    return;
  }
  dst[i] = x[i] + y[i%ky];
}
                            