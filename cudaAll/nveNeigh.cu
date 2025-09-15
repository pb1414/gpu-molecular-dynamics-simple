// md_cuda_nbl.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>

#define N 36000
#define NSTEPS 5000
#define DELT 0.001f
#define RCUT 2.5f
#define SKIN 0.3f
#define RCUT2 (RCUT * RCUT)
#define RCUT_SKIN2 ((RCUT + SKIN) * (RCUT + SKIN))
#define BOX_LENGTH 36.69f
#define PRINT_FREQ 100
#define NBL_UPDATE_FREQ 5
#define MAX_NEIGHBORS 128

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

__global__ void build_neighbor_list(float3* x, int* nbl, int* nbl_count, float box, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    float3 xi = x[i];
    int count = 0;

    for (int j = 0; j < nparticles && count < MAX_NEIGHBORS; ++j) {
        if (i == j) continue;
        float3 dx;
        dx.x = pbc(xi.x - x[j].x, box);
        dx.y = pbc(xi.y - x[j].y, box);
        dx.z = pbc(xi.z - x[j].z, box);

        float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
        if (r2 < RCUT_SKIN2) {
            nbl[i * MAX_NEIGHBORS + count] = j;
            count++;
        }
    }
    nbl_count[i] = count;
}

__global__ void compute_forces_nbl(float3* x, float3* f, int* nbl, int* nbl_count, float box, float* pe, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    float3 xi = x[i];
    float3 fi = make_float3(0.0f, 0.0f, 0.0f);
    float pe_i = 0.0f;

    int ni = nbl_count[i];
    for (int n = 0; n < ni; ++n) {
        int j = nbl[i * MAX_NEIGHBORS + n];
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
    atomicAdd(pe, 0.5f * pe_i);
}

__global__ void verlet_step(float3* x, float3* v, float3* f, float box, float dt, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    v[i].x += 0.5f * dt * f[i].x;
    v[i].y += 0.5f * dt * f[i].y;
    v[i].z += 0.5f * dt * f[i].z;

    x[i].x += v[i].x * dt;
    x[i].y += v[i].y * dt;
    x[i].z += v[i].z * dt;

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

__global__ void compute_kinetic(float3* v, float* ke, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    float vi2 = v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z;
    atomicAdd(ke, 0.5f * vi2);
}

int main() {
    float3 *d_x, *d_v, *d_f;
    float *d_pe, *d_ke;
    int *d_nbl, *d_nbl_count;
    curandState* d_states;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_x, N * sizeof(float3));
    cudaMalloc(&d_v, N * sizeof(float3));
    cudaMalloc(&d_f, N * sizeof(float3));
    cudaMalloc(&d_pe, sizeof(float));
    cudaMalloc(&d_ke, sizeof(float));
    cudaMalloc(&d_states, N * sizeof(curandState));
    cudaMalloc(&d_nbl, N * MAX_NEIGHBORS * sizeof(int));
    cudaMalloc(&d_nbl_count, N * sizeof(int));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    init_positions<<<gridSize, blockSize>>>(d_x, BOX_LENGTH, N);
    init_velocities<<<gridSize, blockSize>>>(d_v, d_states, N);

    float3* h_v = (float3*)malloc(N * sizeof(float3));
    cudaMemcpy(h_v, d_v, N * sizeof(float3), cudaMemcpyDeviceToHost);
    float3 cm = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < N; ++i) {
        cm.x += h_v[i].x;
        cm.y += h_v[i].y;
        cm.z += h_v[i].z;
    }
    cm.x /= N; cm.y /= N; cm.z /= N;
    for (int i = 0; i < N; ++i) {
        h_v[i].x -= cm.x;
        h_v[i].y -= cm.y;
        h_v[i].z -= cm.z;
    }
    cudaMemcpy(d_v, h_v, N * sizeof(float3), cudaMemcpyHostToDevice);
    free(h_v);

    build_neighbor_list<<<gridSize, blockSize>>>(d_x, d_nbl, d_nbl_count, BOX_LENGTH, N);

    for (int step = 0; step < NSTEPS; ++step) {
        if (step % NBL_UPDATE_FREQ == 0) {
            build_neighbor_list<<<gridSize, blockSize>>>(d_x, d_nbl, d_nbl_count, BOX_LENGTH, N);
        }

        cudaMemset(d_pe, 0, sizeof(float));
        cudaMemset(d_ke, 0, sizeof(float));
        zero_forces<<<gridSize, blockSize>>>(d_f, N);
        compute_forces_nbl<<<gridSize, blockSize>>>(d_x, d_f, d_nbl, d_nbl_count, BOX_LENGTH, d_pe, N);
        verlet_step<<<gridSize, blockSize>>>(d_x, d_v, d_f, BOX_LENGTH, DELT, N);
        complete_velocity<<<gridSize, blockSize>>>(d_v, d_f, DELT, N);
        compute_kinetic<<<gridSize, blockSize>>>(d_v, d_ke, N);

        if (step % PRINT_FREQ == 0) {
            float h_pe, h_ke;
            cudaMemcpy(&h_pe, d_pe, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_ke, d_ke, sizeof(float), cudaMemcpyDeviceToHost);
            printf("Step %d | PE: %.4f | KE: %.4f | E: %.4f\n", step, h_pe, h_ke, h_pe + h_ke);
        }
    }

    cudaFree(d_x); cudaFree(d_v); cudaFree(d_f);
    cudaFree(d_pe); cudaFree(d_ke);
    cudaFree(d_states);
    cudaFree(d_nbl); cudaFree(d_nbl_count);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
