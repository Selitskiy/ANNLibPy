#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_kernel(float* c, const float* a, const float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int n = 1024;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(d_c, d_a, d_b, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("h_c[10] = %f\n", h_c[10]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}