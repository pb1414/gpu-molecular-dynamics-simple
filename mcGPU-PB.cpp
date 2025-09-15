// mc-nvt-gpu.cu
// GPU-accelerated Monte Carlo NVT simulation

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>

#define N 500
#define STEP_MAX 2500000
#define RCUT 2.5f
#define RCUT2 (RCUT * RCUT)
#define DR_MAX 0.2f
#define T 0.85f
#define VOL 2500.0f
#define EPSILON 1.0f
#define SIGMA 1.0f
#define PRINT_FREQ (50 * N)
#define BLOCK_SIZE 256

__device__ float pbc(float x, float box_half, float box) {
    if (x > box_half) x -= box;
    else if (x < -box_half) x += box;
    return x;
}

__device__ float U_single_gpu(int i, float3* x, float box, float boxh) {
    float Utotal = 0.0f;
    float3 xi = x[i];

    for (int j = 0; j < N; j++) {
        if (j == i) continue;

        float3 dx;
        dx.x = pbc(xi.x - x[j].x, boxh, box);
        dx.y = pbc(xi.y - x[j].y, boxh, box);
        dx.z = pbc(xi.z - x[j].z, boxh, box);

        float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

        if (r2 < RCUT2 && r2 > 1e-8f) {
            float inv_r2 = 1.0f / r2;
            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;
            Utotal += 4.0f * EPSILON * (inv_r12 - inv_r6);
        }
    }
    return Utotal;
}

__global__ void init_rand(curandState *states, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void mc_step(float3* x, curandState* states, int* accepted, float box, float boxh, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState localState = states[i];
    float3 xi = x[i];
    float3 xold = xi;

    float Uo = U_single_gpu(i, x, box, boxh);

    // Displace
    xi.x += 2.0f * DR_MAX * (curand_uniform(&localState) - 0.5f);
    xi.y += 2.0f * DR_MAX * (curand_uniform(&localState) - 0.5f);
    xi.z += 2.0f * DR_MAX * (curand_uniform(&localState) - 0.5f);

    // Wrap into box
    if (xi.x > boxh) xi.x -= box;
    else if (xi.x < -boxh) xi.x += box;
    if (xi.y > boxh) xi.y -= box;
    else if (xi.y < -boxh) xi.y += box;
    if (xi.z > boxh) xi.z -= box;
    else if (xi.z < -boxh) xi.z += box;

    x[i] = xi;
    float Un = U_single_gpu(i, x, box, boxh);
    float dU = Un - Uo;

    float prob = expf(-beta * dU);
    if ((dU > 0.0f) && (curand_uniform(&localState) > prob)) {
        x[i] = xold;
    } else {
        atomicAdd(accepted, 1);
    }

    states[i] = localState;
}

__global__ void compute_energy(float3* x, float* energy, float box, float boxh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float Ui = U_single_gpu(i, x, box, boxh);
    atomicAdd(energy, 0.5f * Ui);
}

void write_xyz(float3* h_x, int step) {
    FILE* file = fopen("traj085Real.xyz", step == 0 ? "w" : "a");
    fprintf(file, "%d\n\n", N);
    for (int i = 0; i < N; ++i) {
        fprintf(file, "H %f %f %f\n", h_x[i].x, h_x[i].y, h_x[i].z);
    }
    fclose(file);
}

int main() {
    float beta = 1.0f / T;
    float box = pow(VOL, 1.0f / 3.0f);
    float boxh = 0.5f * box;

    float3* h_x = new float3[N];
    float3* d_x;
    float* d_energy;
    int* d_accepted;
    curandState* d_states;

    cudaMalloc(&d_x, N * sizeof(float3));
    cudaMalloc(&d_energy, sizeof(float));
    cudaMalloc(&d_accepted, sizeof(int));
    cudaMalloc(&d_states, N * sizeof(curandState));

    for (int i = 0; i < N; ++i) {
        h_x[i].x = box * (float(rand()) / RAND_MAX) - boxh;
        h_x[i].y = box * (float(rand()) / RAND_MAX) - boxh;
        h_x[i].z = box * (float(rand()) / RAND_MAX) - boxh;
    }

    cudaMemcpy(d_x, h_x, N * sizeof(float3), cudaMemcpyHostToDevice);
    init_rand<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, time(NULL));

    std::ofstream log("log.dat");
    for (int step = 0; step < STEP_MAX; ++step) {
        cudaMemset(d_accepted, 0, sizeof(int));

        mc_step<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_states, d_accepted, box, boxh, beta);

        if (step % PRINT_FREQ == 0) {
            cudaMemset(d_energy, 0, sizeof(float));
            compute_energy<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_energy, box, boxh);
            float Usys;
            int accepted;
            cudaMemcpy(&Usys, d_energy, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&accepted, d_accepted, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_x, d_x, N * sizeof(float3), cudaMemcpyDeviceToHost);

            printf("Step: %d, Usys: %f, Acceptance: %.2f\n", step, Usys, float(accepted) / N);
            log << step << " " << Usys << " " << T << std::endl;
            write_xyz(h_x, step);
        }
    }

    log.close();
    cudaFree(d_x);
    cudaFree(d_states);
    cudaFree(d_energy);
    cudaFree(d_accepted);
    delete[] h_x;
    return 0;
}
