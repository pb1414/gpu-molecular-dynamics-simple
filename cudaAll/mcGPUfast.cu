#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <curand_kernel.h>

#define N 1500  // number of particles
#define STEPS 100000
#define BLOCK_SIZE 256
#define R_CUT 2.5f
#define R_CUT2 (R_CUT * R_CUT)

float box = 19.57f;
float dr_max = 0.2f;
float T = 0.85f;
float beta_inv = 1.0f / T;

// GPU utility: apply periodic boundary conditions
__device__ float pbc(float x, float box) {
    if (x > 0.5f * box) return x - box;
    if (x < -0.5f * box) return x + box;
    return x;
}

// Compute energy of particle i
__device__ float compute_Ui(int i, float3* x, float box) {
    float U = 0.0f;
    float3 xi = x[i];

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float3 dx;
        dx.x = pbc(xi.x - x[j].x, box);
        dx.y = pbc(xi.y - x[j].y, box);
        dx.z = pbc(xi.z - x[j].z, box);
        float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

        if (r2 < R_CUT2 && r2 > 1e-8f) {
            float inv_r2 = 1.0f / r2;
            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;
            U += 4.0f * (inv_r12 - inv_r6);
        }
    }
    return U;
}

// MC step kernel
__global__ void mc_step_kernel(float3* x, float3* xnew, int* accept, curandState* state, float box, float dr_max, float beta_inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState localState = state[i];
    float3 xi = x[i];
    float3 xtrial;

    // Propose move
    xtrial.x = xi.x + dr_max * (2.0f * curand_uniform(&localState) - 1.0f);
    xtrial.y = xi.y + dr_max * (2.0f * curand_uniform(&localState) - 1.0f);
    xtrial.z = xi.z + dr_max * (2.0f * curand_uniform(&localState) - 1.0f);

    xtrial.x = pbc(xtrial.x, box);
    xtrial.y = pbc(xtrial.y, box);
    xtrial.z = pbc(xtrial.z, box);

    float U_old = compute_Ui(i, x, box);
    xnew[i] = xtrial;
    float U_new = compute_Ui(i, xnew, box);

    float dU = U_new - U_old;
    float prob = expf(-beta_inv * dU);

    if (dU <= 0.0f || curand_uniform(&localState) < prob) {
        x[i] = xtrial;
        accept[i] = 1;
    } else {
        xnew[i] = xi;
        accept[i] = 0;
    }

    state[i] = localState;
}

// Kernel to compute total energy
__global__ void compute_total_energy(float3* x, float* energy, float box) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float e = 0.0f;
    float3 xi = x[i];

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float3 dx;
        dx.x = pbc(xi.x - x[j].x, box);
        dx.y = pbc(xi.y - x[j].y, box);
        dx.z = pbc(xi.z - x[j].z, box);
        float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

        if (r2 < R_CUT2 && r2 > 1e-8f) {
            float inv_r2 = 1.0f / r2;
            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;
            e += 4.0f * (inv_r12 - inv_r6);
        }
    }

    atomicAdd(energy, 0.5f * e);
}

// Init RNG
__global__ void setup_rng(curandState* state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
        curand_init(seed, id, 0, &state[id]);
}

// Host main
int main() {
    float3* d_x;
    float3* d_xnew;
    int* d_accept;
    curandState* d_state;

    cudaMalloc(&d_x, N * sizeof(float3));
    cudaMalloc(&d_xnew, N * sizeof(float3));
    cudaMalloc(&d_accept, N * sizeof(int));
    cudaMalloc(&d_state, N * sizeof(curandState));

    // Init positions
    float3* h_x = new float3[N];
    for (int i = 0; i < N; ++i) {
        h_x[i].x = box * ((float)rand() / RAND_MAX - 0.5f);
        h_x[i].y = box * ((float)rand() / RAND_MAX - 0.5f);
        h_x[i].z = box * ((float)rand() / RAND_MAX - 0.5f);
    }
    cudaMemcpy(d_x, h_x, N * sizeof(float3), cudaMemcpyHostToDevice);

    // Init RNG
    setup_rng<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, time(NULL));

    std::ofstream logfile("log.dat");

    int accepted_total = 0;

    for (int step = 0; step < STEPS; ++step) {
        mc_step_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_xnew, d_accept, d_state, box, dr_max, beta_inv);

        int h_accept[N];
        cudaMemcpy(h_accept, d_accept, N * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i)
            accepted_total += h_accept[i];

        if (step % (50 * N) == 0) {
            float* d_energy;
            float h_energy = 0.0f;
            cudaMalloc(&d_energy, sizeof(float));
            cudaMemset(d_energy, 0, sizeof(float));

            compute_total_energy<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_energy, box);
            cudaMemcpy(&h_energy, d_energy, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_energy);

            float acc_ratio = (float)accepted_total / (float)(step + 1);

            printf("Step: %d | U_sys: %.4f | Acceptance Ratio: %.4f\n", step, h_energy, acc_ratio);
            logfile << step << " " << h_energy << " " << T << "\n";
        }
    }

    logfile.close();
    cudaFree(d_x);
    cudaFree(d_xnew);
    cudaFree(d_accept);
    cudaFree(d_state);
    delete[] h_x;

    return 0;
}
