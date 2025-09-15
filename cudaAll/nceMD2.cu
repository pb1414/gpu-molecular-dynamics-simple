// Molecular Dynamics using Velocity Verlet in CUDA
// Matches initial setup (box volume, velocities) with CPU version

#include <cstdio>
#include <cmath>
#include <curand_kernel.h>

#define NPARTICLES 2000
#define VOLUME 2700.0f
#define BOX_LENGTH cbrtf(VOLUME)
#define DT 0.001f
#define RCUT 2.5f
#define RCUT2 (RCUT * RCUT)
#define NSTEPS 1000
#define MASS 1.0f
#define KB 1.0f
#define INIT_TEMP 0.5f

__host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 apply_pbc(float3 dr, float box) {
    if (dr.x > 0.5f * box) dr.x -= box;
    if (dr.x < -0.5f * box) dr.x += box;
    if (dr.y > 0.5f * box) dr.y -= box;
    if (dr.y < -0.5f * box) dr.y += box;
    if (dr.z > 0.5f * box) dr.z -= box;
    if (dr.z < -0.5f * box) dr.z += box;
    return dr;
}

__global__ void init_positions(float3* x, float box, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    int n = ceilf(cbrtf((float)nparticles));
    float spacing = box / n;

    int ix = i / (n * n);
    int iy = (i / n) % n;
    int iz = i % n;

    x[i] = make_float3(
        ix * spacing + 0.05f * spacing,
        iy * spacing + 0.05f * spacing,
        iz * spacing + 0.05f * spacing
    );
}

__global__ void init_velocities(float3* v, float3* vcm, curandState* states, float stddev, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    curand_init(1234, i, 0, &states[i]);
    float3 vel;
    vel.x = curand_normal(&states[i]) * stddev;
    vel.y = curand_normal(&states[i]) * stddev;
    vel.z = curand_normal(&states[i]) * stddev;

    v[i] = vel;

    atomicAdd(&vcm->x, vel.x);
    atomicAdd(&vcm->y, vel.y);
    atomicAdd(&vcm->z, vel.z);
}

__global__ void remove_cm_velocity(float3* v, float3 vcm, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    vcm.x /= nparticles;
    vcm.y /= nparticles;
    vcm.z /= nparticles;

    v[i].x -= vcm.x;
    v[i].y -= vcm.y;
    v[i].z -= vcm.z;
}

__global__ void zero_forces(float3* f, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;
    f[i] = make_float3(0, 0, 0);
}

__global__ void compute_forces(float3* x, float3* f, float* pe, float box, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    float3 xi = x[i];
    float3 fi = make_float3(0, 0, 0);
    float local_pe = 0.0f;

    for (int j = 0; j < nparticles; ++j) {
        if (i == j) continue;
        float3 dx = apply_pbc(xi - x[j], box);
        float r2 = dx.x*dx.x + dx.y*dx.y + dx.z*dx.z;

        if (r2 < RCUT2) {
            float r2inv = 1.0f / r2;
            float r6inv = r2inv * r2inv * r2inv;
            float fmag = 48.0f * r6inv * (r6inv - 0.5f) * r2inv;
            fi.x += fmag * dx.x;
            fi.y += fmag * dx.y;
            fi.z += fmag * dx.z;
            local_pe += 4.0f * r6inv * (r6inv - 1.0f);
        }
    }
    f[i] = fi;
    atomicAdd(pe, local_pe * 0.5f);
}

__global__ void integrate(float3* x, float3* v, float3* f, float box, float dt, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    v[i].x += 0.5f * dt * f[i].x;
    v[i].y += 0.5f * dt * f[i].y;
    v[i].z += 0.5f * dt * f[i].z;

    x[i].x += dt * v[i].x;
    x[i].y += dt * v[i].y;
    x[i].z += dt * v[i].z;

    // Apply periodic boundaries
    x[i].x = fmodf(x[i].x + box, box);
    x[i].y = fmodf(x[i].y + box, box);
    x[i].z = fmodf(x[i].z + box, box);
}

__global__ void finish_velocity(float3* v, float3* f, float dt, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    v[i].x += 0.5f * dt * f[i].x;
    v[i].y += 0.5f * dt * f[i].y;
    v[i].z += 0.5f * dt * f[i].z;
}

__global__ void compute_kinetic(float3* v, float* ke, int nparticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nparticles) return;

    float vel2 = v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z;
    atomicAdd(ke, 0.5f * MASS * vel2);
}

int main() {
    printf("Starting MD...\n");
    float3 *d_x, *d_v, *d_f;
    float *d_ke, *d_pe;
    curandState *d_states;

    cudaMalloc(&d_x, NPARTICLES * sizeof(float3));
    cudaMalloc(&d_v, NPARTICLES * sizeof(float3));
    cudaMalloc(&d_f, NPARTICLES * sizeof(float3));
    cudaMalloc(&d_ke, sizeof(float));
    cudaMalloc(&d_pe, sizeof(float));
    cudaMalloc(&d_states, NPARTICLES * sizeof(curandState));

    float3 zero = {0,0,0};
    float3 *d_vcm;
    cudaMalloc(&d_vcm, sizeof(float3));
    cudaMemcpy(d_vcm, &zero, sizeof(float3), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (NPARTICLES + blockSize - 1) / blockSize;

    init_positions<<<gridSize, blockSize>>>(d_x, BOX_LENGTH, NPARTICLES);
    init_velocities<<<gridSize, blockSize>>>(d_v, d_vcm, d_states, sqrtf(INIT_TEMP), NPARTICLES);
    cudaDeviceSynchronize();
    remove_cm_velocity<<<gridSize, blockSize>>>(d_v, *d_vcm, NPARTICLES);
    

    for (int step = 0; step < NSTEPS; ++step) {
        cudaMemset(d_pe, 0, sizeof(float));
        cudaMemset(d_ke, 0, sizeof(float));

        zero_forces<<<gridSize, blockSize>>>(d_f, NPARTICLES);
        compute_forces<<<gridSize, blockSize>>>(d_x, d_f, d_pe, BOX_LENGTH, NPARTICLES);
        integrate<<<gridSize, blockSize>>>(d_x, d_v, d_f, BOX_LENGTH, DT, NPARTICLES);

        zero_forces<<<gridSize, blockSize>>>(d_f, NPARTICLES);
        compute_forces<<<gridSize, blockSize>>>(d_x, d_f, d_pe, BOX_LENGTH, NPARTICLES);
        finish_velocity<<<gridSize, blockSize>>>(d_v, d_f, DT, NPARTICLES);

        compute_kinetic<<<gridSize, blockSize>>>(d_v, d_ke, NPARTICLES);



        float h_ke, h_pe;
        cudaMemcpy(&h_ke, d_ke, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pe, d_pe, sizeof(float), cudaMemcpyDeviceToHost);

        if (step % 50 == 0)
            printf("Step %d: E = %f PE = %f KE = %f T = %f\n", step, h_ke + h_pe, h_pe, h_ke, 2*h_ke/(3.0f*NPARTICLES));
    }

    cudaFree(d_x); cudaFree(d_v); cudaFree(d_f);
    cudaFree(d_pe); cudaFree(d_ke); cudaFree(d_states); cudaFree(d_vcm);
    return 0;
}
