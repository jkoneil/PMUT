// tr_kernel.cu
// CUDA kernel for time-reversal photoacoustic reconstruction (voxel-gather).
// One thread = one voxel. Loaded and compiled at runtime by tr_gpu.py (CuPy),
// and reusable unchanged in a future compiled (nvcc + pybind11) build.

extern "C" __global__
void tr_gather(const float* __restrict__ Receive,   // [nSens][nTm1], derivative of signal
               const float* __restrict__ sx,        // sensor x (mm)
               const float* __restrict__ sy,        // sensor y (mm)
               float* __restrict__ img,             // [nz][nx][ny], y fastest
               int nSens, int nTm1, int nz, int nx, int ny,
               float x0, float y0, float z0,        // coordinate of voxel (0,0,0), mm
               float dx, float dy, float dz,        // voxel size, mm
               float inv_cdt,                       // 1 / (co * dt)
               float tanThetaMax)
{
    int iy = blockIdx.x * blockDim.x + threadIdx.x;  // consecutive threads -> consecutive y (coalesced)
    int ix = blockIdx.y;
    int iz = blockIdx.z;
    if (iy >= ny) return;

    size_t out_idx = ((size_t)iz * nx + ix) * ny + iy;

    float x = x0 + ix * dx;
    float y = y0 + iy * dy;
    float z = z0 + iz * dz;

    if (z <= 0.0f) { img[out_idx] = 0.0f; return; }

    float rhoMax = z * tanThetaMax;
    float rhoMax2 = rhoMax * rhoMax;
    float sum = 0.0f;

    for (int s = 0; s < nSens; s++) {
        float ddx = x - sx[s];
        float ddy = y - sy[s];
        float rho2 = ddx * ddx + ddy * ddy;
        if (rho2 > rhoMax2) continue;                 // outside this sensor's +/- ThetaMax cone

        float R = sqrtf(rho2 + z * z);                // distance sensor -> voxel
        // CPU code uses l = (t+1)*dt*co  =>  t = R/(co*dt) - 1
        float tf = R * inv_cdt - 1.0f;
        int t0 = (int)floorf(tf);
        if (t0 < 0 || t0 >= nTm1 - 1) continue;
        float w = tf - (float)t0;

        const float* r = Receive + (size_t)s * nTm1;
        sum += R * ((1.0f - w) * r[t0] + w * r[t0 + 1]);   // linear interpolation
    }
    img[out_idx] = sum;
}
