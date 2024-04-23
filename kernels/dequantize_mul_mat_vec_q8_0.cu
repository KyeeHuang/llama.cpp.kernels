#define QK8_0 32
#define QR8_0 1
#define WARP_SIZE 32

typedef struct {
  half  d;
  int8_t qs[QK8_0];
} block_q8_0;

static void dequantize_mul_mat_vec_q8_0_cuda(const void* vx, const dfloat * y, 
            float *dst, const int ncols, const int nrows, cudaStream_t stream) 
{
  GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(1, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1); 
  dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>
    <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows); 
}

static __device__ __forceinline__ void dequantize_q8_0(const void *vx, const int ib, const int iqs, dfloat2 & v) {
  // 均匀对称量化
  // dequantize is int8 * scale
  const block_q8_0* x = (const block_q8_0*) vx;
  const dfloat2 d = x[ib].d; // scale
  
  v.x = x[ib].qs[iqs+0];
  v.y = x[ib].qs[iqs+1];

#ifdef GGML_CUDA_FP16
  // FP16
  v = __hmul2(v, {d, d});
#else
  // FP32
  v.x *= d;
  v.y *= d;
#endif
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat *__restrict__ y,
                          float* __restrict__ dst, const int ncols, const int nrows)
{
  // qk = quantized weights per x block
  // qr = number of quantized weights per data value in x block
  
  const int row = blockIdx.y * blockDim.y + threadIdx.x;
  
  if (row >= nrows) return;

  const int tid = threadIdx.x;
  const int iter_stride = 2 * GGML_CUDA_DMMV_X; // 2*32
  const int vals_per_iter = iter_stride / WARP_SIZE;
  const int y_offset = qr == 1 ? 1 : qk/2;

#ifdef GGML_CUDA_FP16
  half2 tmp = {0.0f, 0.0f};
#else 
  float tmp = 0.0f;
#endif
  // 32 threads process 4096-set data
  
  for (int i =0; i < ncols; i += iter_stride) {
    const int col = i + vals_per_iter * tid;
    const int ib = (row * ncols + col) / qk;
    const int iqs = (col % qk) / qr;
    const int iybs = col - col % qk;
    
    for (int j = 0; j < vals_per_iter; j+= 2) {
      // 2 vals per j iter
      
      // dequantize
      // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
      dfloat2 v;
      dequantize_kernel(vx, ib, iqs + j / qr, v);
      
#ifdef GGML_CUDA_FP16
      tmp += __hmul2(v, {
        y[iybs + iqs + j/qr + 0],
        y[iybs + iqs + j/qr + y_offset]
      });
#else
      tmp += v.x * y[iybs + iqs + j / qr + 0];
      tmp += v.y * y[iybs + iqs + j / qr + y_offset];
#endif
    }
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
  }

  if (tid == 0) {
#ifdef GGML_CUDA_FP16
    dst[row] = tmp.x + tmp.y;
#else
    dst[row] = tmp; 
#endif
  }
}