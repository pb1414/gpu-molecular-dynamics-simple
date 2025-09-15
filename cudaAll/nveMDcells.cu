#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1000              // Number of particles
#define RCUT 2.5f           // Cutoff radius
#define BOXSIZE 20.0f       // Simulation box length
#define NUM_STEPS 1000      // Number of MD steps
#define DT 0.005f           // Timestep

__device__ int get_cell_index(int ix, int iy, int iz, int cells_per_dim) {
    ix = (ix + cells_per_dim) % cells_per_dim;
    iy = (iy + cells_per_dim) % cells_per_dim;
    iz = (iz + cells_per_dim) % cells_per_dim;
    return ix + iy * cells_per_dim + iz * cells_per_dim * cells_per_dim;
}

__global__ void compute_forces(int *cell_head, int *cell_list, float *x, float *f, int n_particles, int cells_per_dim, float rcut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    float xi = x[idx * 3];
    float yi = x[idx * 3 + 1];
    float zi = x[idx * 3 + 2];

    f[idx * 3] = 0.0f;
    f[idx * 3 + 1] = 0.0f;
    f[idx * 3 + 2] = 0.0f;

    int ix = static_cast<int>(xi / rcut);
    int iy = static_cast<int>(yi / rcut);
    int iz = static_cast<int>(zi / rcut);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int jx = ix + dx;
                int jy = iy + dy;
                int jz = iz + dz;
                int neighbor_cell = get_cell_index(jx, jy, jz, cells_per_dim);

                int j = cell_head[neighbor_cell];
                while (j != -1) {
                    if (j != idx) {
                        float dx = x[j * 3] - xi;
                        float dy = x[j * 3 + 1] - yi;
                        float dz = x[j * 3 + 2] - zi;

                        // Minimum image convention
                        dx -= BOXSIZE * roundf(dx / BOXSIZE);
                        dy -= BOXSIZE * roundf(dy / BOXSIZE);
                        dz -= BOXSIZE * roundf(dz / BOXSIZE);

                        float r2 = dx*dx + dy*dy + dz*dz;
                        if (r2 < rcut * rcut && r2 > 1e-12f) {
                            float r2inv = 1.0f / r2;
                            float r6 = r2inv * r2inv * r2inv;
                            float ff = 48.0f * r6 * (r6 - 0.5f) * r2inv;

                            atomicAdd(&f[idx * 3], ff * dx);
                            atomicAdd(&f[idx * 3 + 1], ff * dy);
                            atomicAdd(&f[idx * 3 + 2], ff * dz);
                        }
                    }
                    j = cell_list[j];
                }
            }
        }
    }
}

void init_cell_list(float rcut, float *x, int *cell_head, int *cell_list, int n_particles) {
    int cells_per_dim = static_cast<int>(BOXSIZE / rcut);
    int total_cells = cells_per_dim * cells_per_dim * cells_per_dim;

    for (int i = 0; i < total_cells; i++) cell_head[i] = -1;

    for (int i = 0; i < n_particles; i++) {
        float xi = x[i * 3];
        float yi = x[i * 3 + 1];
        float zi = x[i * 3 + 2];

        int ix = static_cast<int>(xi / rcut);
        int iy = static_cast<int>(yi / rcut);
        int iz = static_cast<int>(zi / rcut);

        int cell = ix + iy * cells_per_dim + iz * cells_per_dim * cells_per_dim;
        cell_list[i] = cell_head[cell];
        cell_head[cell] = i;
    }
}

void run_md_simulation(float *x, float *f, int n_particles, float rcut, int num_steps) {
    float *d_x, *d_f;
    int *d_cell_head, *d_cell_list;
    int cells_per_dim = static_cast<int>(BOXSIZE / rcut);
    int total_cells = cells_per_dim * cells_per_dim * cells_per_dim;

    cudaMalloc(&d_x, 3 * n_particles * sizeof(float));
    cudaMalloc(&d_f, 3 * n_particles * sizeof(float));
    cudaMalloc(&d_cell_head, total_cells * sizeof(int));
    cudaMalloc(&d_cell_list, n_particles * sizeof(int));

    cudaMemcpy(d_x, x, 3 * n_particles * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(128);
    dim3 grid_size((n_particles + block_size.x - 1) / block_size.x);

    for (int step = 0; step < num_steps; step++) {
        if (step % 100 == 0) {
            std::cout << "Step " << step << " / " << num_steps << std::endl;
        }

        int *cell_head = new int[total_cells];
        int *cell_list = new int[n_particles];

        init_cell_list(rcut, x, cell_head, cell_list, n_particles);

        cudaMemcpy(d_cell_head, cell_head, total_cells * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cell_list, cell_list, n_particles * sizeof(int), cudaMemcpyHostToDevice);

        compute_forces<<<grid_size, block_size>>>(d_cell_head, d_cell_list, d_x, d_f, n_particles, cells_per_dim, rcut);
        cudaDeviceSynchronize();

        // Integration step (simplified: no velocities shown)
        cudaMemcpy(f, d_f, 3 * n_particles * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_particles; i++) {
            for (int j = 0; j < 3; j++) {
                x[i * 3 + j] += DT * f[i * 3 + j];
                x[i * 3 + j] = fmodf(x[i * 3 + j] + BOXSIZE, BOXSIZE);
            }
        }

        cudaMemcpy(d_x, x, 3 * n_particles * sizeof(float), cudaMemcpyHostToDevice);
        delete[] cell_head;
        delete[] cell_list;
    }

    cudaFree(d_x);
    cudaFree(d_f);
    cudaFree(d_cell_head);
    cudaFree(d_cell_list);
}

int main() {
    float *x = new float[3 * N];
    float *f = new float[3 * N];

    for (int i = 0; i < N; i++) {
        x[3 * i]     = static_cast<float>(rand()) / RAND_MAX * BOXSIZE;
        x[3 * i + 1] = static_cast<float>(rand()) / RAND_MAX * BOXSIZE;
        x[3 * i + 2] = static_cast<float>(rand()) / RAND_MAX * BOXSIZE;
    }

    run_md_simulation(x, f, N, RCUT, NUM_STEPS);

    delete[] x;
    delete[] f;

    return 0;
}