#define QK8_1 32

typedef struct {
  half2 ds;
  int8_t qs[QK8_0];
} block_q8_1;

static __global__ void quantize_q8_1(const float* __restrict__ x, void* __restrict__ vy, const int kx,
                            const int kx_padded) 
{
  const int ix = blockDim.x * blockIdx.x + threadIdx.x; // 0-4096
  
  if (ix >= kx_padded) return;

  const int iy = blockDimx.y * blockIdx.y + threadIdx.y; // 0
  const int i_padded = iy * ky_padded + ix;
  block_q8_1* y = (block_q8_1*) vy;

  const int ib = i_padded / QK8_1; // block index
  const int iqs = i_padded % QK8_1; // quant index

  const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;
  
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
    sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
  }

  // q = round(clip(r_i / scale, Q_{min}, Q_{max}))
  // scale = fmax - fmin / qmax - qmin
  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) return;
  
  y[ib].ds.x = d;
  y[ib].ds.y = sum;
}