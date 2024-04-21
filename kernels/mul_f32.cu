#define CUDA_MUL_BLOCK_SIZE 256

static void mul_f32_cuda(const float* x, const float* y, float* dst, 
                         const int kx, const int ky, cudaStream_t stream)
{
  const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
  mul_f32<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}

// mul gamma
static __global__ void mul_f32(const float* x, const float* y, float * dst,
                               const int kx, const int ky)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i >= kx) return;

  dst[i] = x[i] * y[i % ky];
}