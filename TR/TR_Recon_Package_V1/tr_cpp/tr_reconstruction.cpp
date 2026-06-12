#include <pybind11/pybind11.h> // tools to connect C++ to Python
#include <pybind11/numpy.h> // to pass numpy arrays between Python and C++
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
namespace py = pybind11;

void reconstruct_sensor_inplace(
    py::array_t<double> PASignal,
    py::array_t<double> coord,
    py::array_t<double> zo, 
    py::array_t<double> xo, 
    py::array_t<double> yo,
    double co, double dt, 
    double dxo, 
    double dyo, 
    double dzo,
    double zmin, 
    double xmax, 
    double ymax, 
    double ThetaMax,
    int XY,
    vector<double>& ImgTemp,
    vector<float>&  NumPts,
    vector<double>& ImgLocal)
{
    auto sig = PASignal.unchecked<2>();
    auto crd = coord.unchecked<2>();

    int nz = zo.size(), nx = xo.size(), ny = yo.size();
    int nT = PASignal.shape(0);
    int nVox = nz * nx * ny;

    fill(ImgTemp.begin(), ImgTemp.end(), 0.0);
    fill(NumPts.begin(),  NumPts.end(),  0.0f);

    double Xsens = crd(XY, 0) * 0.001;
    double Ysens = crd(XY, 1) * 0.001;

    vector<double> Receive(nT - 1);
    for (int i = 0; i < nT - 1; i++)
        Receive[i] = sig(i+1, XY) - sig(i, XY);

    for (int t = 0; t < (int)Receive.size(); t++) // loop over time steps
    {
        double l = (t + 1.0) * dt * co; // distance wave has travelled
        if (l == 0.0) continue; // skip if distance is 0

        double dTheta = M_PI / (l * M_PI / dxo);
        double dphi   = dTheta;

        int nPhi = (int)(2 * M_PI / dphi);
        vector<double> cosPhi(nPhi), sinPhi(nPhi); // to store precomputed trig values
        for (int ip = 0; ip < nPhi; ip++) 
        {
            cosPhi[ip] = cos(ip * dphi);
            sinPhi[ip] = sin(ip * dphi);
        }

        for (double theta = -ThetaMax; theta <= ThetaMax + dTheta; theta += dTheta) // loop over elevation angles 
        {
            double cosT = cos(theta), sinT = sin(theta);
            double zval = l * cosT - zmin;
            if (zval < 0) continue; // skip if outside the volume

            int zind = (int)round(zval / dzo);
            if (zind < 0 || zind >= nz) continue;

            double lsinT = l * sinT;

            for (int ip = 0; ip < nPhi; ip++) 
            {
                double xval = lsinT * cosPhi[ip] + Xsens + xmax;
                double yval = lsinT * sinPhi[ip] + Ysens + ymax;

                int xind = (int)round(xval / dxo);
                int yind = (int)round(yval / dyo);

                if (xind < 0 || xind >= nx) continue;
                if (yind < 0 || yind >= ny) continue;

                int idx = zind * nx * ny + xind * ny + yind;
                ImgTemp[idx] += l * Receive[t];
                NumPts[idx]  += 1.0f;
            }
        }
    }

    // Accumulate into ImgLocal
    for (int i = 0; i < nVox; i++)
        ImgLocal[i] += ImgTemp[i] / (NumPts[i] + 1.0f);
}

// Returns numpy array back to python
py::array_t<double> time_reversal_reconstruction(
    py::array_t<double> PASignal,
    py::array_t<double> coord,
    py::array_t<double> zo, py::array_t<double> xo, py::array_t<double> yo,
    double co, double dt, double dxo, double dyo, double dzo,
    double zmin, double xmax, double ymax, double ThetaMax)
{
    int nSensors = coord.shape(0);
    int nz = zo.size(), nx = xo.size(), ny = yo.size();
    int nVox = nz * nx * ny;

    vector<double> Img(nVox, 0.0);

    #ifdef _OPENMP
    #pragma omp parallel // launch threads, everything in this block runs in parallel
    {
        // Each thread gets own buffers
        vector<double> ImgLocal(nVox, 0.0);
        vector<double> ImgTemp(nVox, 0.0);
        vector<float>  NumPts(nVox,  0.0f);

        #pragma omp for schedule(dynamic)
        // split sensor loop across threads
        for (int XY = 0; XY < nSensors; XY++) {
            reconstruct_sensor_inplace(
                PASignal, coord, zo, xo, yo,
                co, dt, dxo, dyo, dzo, zmin, xmax, ymax, ThetaMax,
                XY, ImgTemp, NumPts, ImgLocal);
        }
        #pragma omp critical // only one thread at a time runs to add local result
        for (int i = 0; i < nVox; i++)
            Img[i] += ImgLocal[i];
    }
    #else // Same thing, no threading
    {
        vector<double> ImgLocal(nVox, 0.0);
        vector<double> ImgTemp(nVox, 0.0);
        vector<float>  NumPts(nVox,  0.0f);
        for (int XY = 0; XY < nSensors; XY++) {
            reconstruct_sensor_inplace(
                PASignal, coord, zo, xo, yo,
                co, dt, dxo, dyo, dzo, zmin, xmax, ymax, ThetaMax,
                XY, ImgTemp, NumPts, ImgLocal);
        }
        for (int i = 0; i < nVox; i++)
            Img[i] = ImgLocal[i];
    }
    #endif

    // Convert flat vector back into 3D numpy array
    auto result = py::array_t<double>({nz, nx, ny});
    auto r = result.mutable_unchecked<3>();
    for (int iz = 0; iz < nz; iz++)
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy < ny; iy++)
                r(iz, ix, iy) = Img[iz*nx*ny + ix*ny + iy];
    return result;
}

PYBIND11_MODULE(tr_cpp, m) // Python module named tr_cpp
{
    m.def("time_reversal_reconstruction", &time_reversal_reconstruction);
    #ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    m.attr("num_threads") = omp_get_max_threads();
    #else
    m.attr("openmp_enabled") = false;
    m.attr("num_threads") = 1;
    #endif
}