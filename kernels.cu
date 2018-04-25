

__global__ void testKernel(double *xs, double *b){
    b[blockIdx.x] = xs[blockIdx.x];
}
