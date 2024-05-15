// fp32 -> fp32
static __device__ void cpy_1_f32_f32 (const char *cxi, char *cdsti) {
  const float * xi = (const float*) cxi;
  float *dsti = (float *) cdsti;
  
  *dsti = *xi;
}

// fp32 -> fp16
static __device__ vodi cpy_1_f32_f16 (const char *cxi, char *cdsti) {
  const float *xi = (const float*) cxi;
  half *dsti = (float *) cdsti;
  
  *dsti = __float2half(*xi);
}

template <cpy_kernel_t cpy_1>
static __global__ void cpy_f32_f16 (const char* cx, char * cdst, const int ne,
                                    const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
                                    const int ne10, const int ne11, const int nb10, const int nb11, const int nb12)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i >= ne) return;

  const int i02 = i / (ne00 * ne01); // index of ne02
  const int i01 = (i - i02*ne00*ne01) / ne00; // index of ne01
  const int i00 = i - i02*ne01*ne00 - i01*ne00; // index of ne00
  const int x_offset = i00*nb00 + i01*nb01 + i02*nb02; // offset
  
  const int i12 = i / (ne10 * ne11); // dst
  const int i11 = (i - i12*ne10*ne00) / ne10;
  const int i10 = i - i12*ne10*ne11 - i11*ne10;
  const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12;
  
  cpy_1(cx + x_offset, cdst + dst_offset);
}