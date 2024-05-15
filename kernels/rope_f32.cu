static __global__ void rope_f32 (const float *x, float *dst, const int ncols,
                                 const float p0, const float p_delta,
                                 const int p_delta_rows, const float theta_scale)
{
  const int col = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
  if (col >= ncols) return;
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int i = row * ncols + col;
  
  const float theta = (p0 + p_delta * (row / p_delta_rows)) *
                      powf(theta_scale, col/2);
  const float sin_theta = sinf(theta);
  const float cos_theta = cosf(theta);
  
  const float x0 = x[i+0];
  const float x1 = x[i+1];
  
  dst[i+0] = x0 * cos_theta - x1 * sin_theta;
  dst[i+1] = x0 * sin_theta + x1 * cos_theta;
}