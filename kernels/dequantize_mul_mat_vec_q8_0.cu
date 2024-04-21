#define QK8_0 32
#define QR8_0 1

static void dequantize_mul_mat_vec_q8_0_cuda(const void* vx, const dfloat * y, 
            float *dst, const int ncols, const int nrows, cudaStream_t stream) 
{
  GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
  
              
}