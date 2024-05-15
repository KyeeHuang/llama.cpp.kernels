// gridDim = {1, 32, 1}, blockDim = {32, 1, 1}
static __global__ void soft_max_f32(const float* x, float* dst, const int ncols) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int block_size = blockDim.x;
  const int tid = threadIdx.x;
  float tmp = 0.0;
  for (int block_start = 0; block_start < ncols; block_start += block_size) {
    const int col = block_start + tid;
    if (col >= ncols) {
      break;
    }
    const int i = row * ncols + col;
    const float val = expf(x[i]);
    tmp += val;
    dst[i] = val;
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
  }

  for (int block_start = 0; block_start < ncols; block_start += block_size) {
    const int col = block_start + tid;
    if (col >= ncols) {
      break;
    }
    const int i = row * ncols + col;
    dst[i] /= tmp;
  }
}