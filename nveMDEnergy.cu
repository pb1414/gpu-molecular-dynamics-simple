// md_cuda.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>

#define N 36000
#define NSTEPS 1000
#define DELT 0.001f
#define RCUT 2.5f
#define RCUT2 (RCUT * RCUT)
#define BOX_LENGTH 36.69f
#define PRINT_FREQ 100

__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ __host__ inline float pbc(float x, float box) {
    if (x > 0.5f * box) x -= box;
    else if (x < -0.5f * box) x += box;
    return x;
}

__global__ void init_positions(float3* x, float box, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    int n = ceilf(cbrtf((float)nparticles));
    float spacing = box / n;

    int ix = i / (n * n);
    int iy = (i / n) % n;
    int iz = i % n;

    x[i].x = ix * spacing + 0.01f;
    x[i].y = iy * spacing + 0.01f;
    x[i].z = iz * spacing + 0.01f;
}

__global__ void init_velocities(float3* v, curandState* states, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    curand_init(1234, i, 0, &states[i]);
    float sigma = 0.01f;
    v[i].x = curand_normal(&states[i]) * sigma;
    v[i].y = curand_normal(&states[i]) * sigma;
    v[i].z = curand_normal(&states[i]) * sigma;
}

__global__ void zero_forces(float3* f, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nparticles)
        f[i] = make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void compute_forces(float3* x, float3* f, float box, double* pe, int nparticles){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    float3 xi = x[i];
    float3 fi = make_float3(0.0f, 0.0f, 0.0f);
    double pe_i = 0.0;

    for (int j = 0; j < nparticles; ++j) {
        if (i == j) continue;

        float3 dx;
        dx.x = pbc(xi.x - x[j].x, box);
        dx.y = pbc(xi.y - x[j].y, box);
        dx.z = pbc(xi.z - x[j].z, box);

        float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
        if (r2 < RCUT2) {
            float inv_r2 = 1.0f / r2;
            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;

            float fmag = 24.0f * inv_r2 * (2.0f * inv_r12 - inv_r6);
            fi.x += fmag * dx.x;
            fi.y += fmag * dx.y;
            fi.z += fmag * dx.z;

            pe_i += 4.0f * (inv_r12 - inv_r6);
        }
    }

    f[i] = fi;
    atomicAdd_double(pe, 0.5 * pe_i);
}

__global__ void verlet_step(float3* x, float3* v, float3* f, float box, float dt, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    // Half velocity step
    v[i].x += 0.5f * dt * f[i].x;
    v[i].y += 0.5f * dt * f[i].y;
    v[i].z += 0.5f * dt * f[i].z;

    // Position update
    x[i].x += v[i].x * dt;
    x[i].y += v[i].y * dt;
    x[i].z += v[i].z * dt;

    // Apply PBCs
    x[i].x = fmodf(x[i].x + box, box);
    x[i].y = fmodf(x[i].y + box, box);
    x[i].z = fmodf(x[i].z + box, box);
}

__global__ void complete_velocity(float3* v, float3* f, float dt, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nparticles) {
        v[i].x += 0.5f * dt * f[i].x;
        v[i].y += 0.5f * dt * f[i].y;
        v[i].z += 0.5f * dt * f[i].z;
    }
}

__global__ void compute_kinetic(float3* v, double* ke, int nparticles){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    double vi2 = v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z;
    atomicAdd_double(ke, 0.5 * vi2);
}

int main() {
    float3 *d_x, *d_v, *d_f;
    double *d_pe, *d_ke;
    curandState* d_states;
    auto start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_x, N * sizeof(float3));
    cudaMalloc(&d_v, N * sizeof(float3));
    cudaMalloc(&d_f, N * sizeof(float3));
    cudaMalloc(&d_states, N * sizeof(curandState));
    cudaMalloc(&d_pe, sizeof(double));
    cudaMalloc(&d_ke, sizeof(double));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Initialization
    init_positions<<<gridSize, blockSize>>>(d_x, BOX_LENGTH, N);
    init_velocities<<<gridSize, blockSize>>>(d_v, d_states, N);

    // Remove center-of-mass drift
    float3* h_v = (float3*)malloc(N * sizeof(float3));
    cudaMemcpy(h_v, d_v, N * sizeof(float3), cudaMemcpyDeviceToHost);
    
    // Compute COM velocity
    float3 cm = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < N; ++i) {
        cm.x += h_v[i].x;
        cm.y += h_v[i].y;
        cm.z += h_v[i].z;
    }
    cm.x /= N;
    cm.y /= N;
    cm.z /= N;
    
    // Subtract COM velocity from each particle
    for (int i = 0; i < N; ++i) {
        h_v[i].x -= cm.x;
        h_v[i].y -= cm.y;
        h_v[i].z -= cm.z;
    }
    cudaMemcpy(d_v, h_v, N * sizeof(float3), cudaMemcpyHostToDevice);
    free(h_v);
    
    for (int step = 0; step < NSTEPS; ++step) {
        cudaMemset(d_pe, 0, sizeof(double));
        cudaMemset(d_ke, 0, sizeof(double));
        

        zero_forces<<<gridSize, blockSize>>>(d_f, N);
        compute_forces<<<gridSize, blockSize>>>(d_x, d_f, BOX_LENGTH, d_pe, N);
        verlet_step<<<gridSize, blockSize>>>(d_x, d_v, d_f, BOX_LENGTH, DELT, N);
        complete_velocity<<<gridSize, blockSize>>>(d_v, d_f, DELT, N);
        compute_kinetic<<<gridSize, blockSize>>>(d_v, d_ke, N);

        if (step % PRINT_FREQ == 0) {
            double h_pe, h_ke;
            cudaMemcpy(&h_pe, d_pe, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_ke, d_ke, sizeof(double), cudaMemcpyDeviceToHost);
            printf("Step %d | PE: %.8f | KE: %.8f | E: %.8f\n", step, h_pe, h_ke, h_pe + h_ke);
        }
    }

    cudaFree(d_x);
    cudaFree(d_v);
    cudaFree(d_f);
    cudaFree(d_states);
    cudaFree(d_pe);
    cudaFree(d_ke);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
