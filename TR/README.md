PMUT Photoacoustic Reconstruction using Time Reversal algorithm in C++, exposed to Python via pybind11


### Python Dependencies

numpy
scipy
matplotlib
scikit-image
pybind11
joblib


Install with:
pip install numpy scipy matplotlib scikit-image pybind11 joblib

### C++ Build Dependencies

**Mac:**
brew install libomp

**Windows:**
If using a different compiler (e.g. MinGW or GCC), you may need to install OpenMP separately. 

---

## Building the C++ Extension

The C++ extension must be compiled before running the reconstruction. From the `TR_Recon_Package_V1` directory:

**Mac:**
ARCHFLAGS="-arch arm64" python3 setup.py build_ext --inplace

**Windows:**
python setup.py build_ext --inplace

This produces a compiled `.so` file (Mac) or `.pyd` file (Windows) that Python imports automatically.

To verify the build was successful:
python3 -c "import tr_cpp; print('OpenMP:', tr_cpp.openmp_enabled, 'Threads:', tr_cpp.num_threads)"


---

## Running the Reconstruction

python3 TRHexArray3D_para.py --data "filename"

This will:
1. Load the PA signal dataset from `Processed/filename.mat`
2. Run the time reversal reconstruction
3. Display using 3D slicer

---

## Project Structure

TR_Recon_Package_V1/
├── TRHexArray3D_para.py          # main script
├── TR_Reconstruction.py          # Python wrapper for C++ extension
├── setup.py                      # builds the C++ extension
├── coordsHexagonal0p5mm.txt      # sensor coordinates
├── tr_cpp/
│   └── tr_reconstruction.cpp     # C++ reconstruction implementation
└── Processed/
    └── d3_PMUT_61_channels_400.mat  # input data

---

## Implementation Notes

The reconstruction kernel is written in C++ for performance. Optimizations over the original Python version:

- Trigonometric values are precomputed once per time step rather than recalculated on every inner loop iteration
- OpenMP parallelizes the reconstruction across all available CPU cores, with per-thread accumulation buffers to avoid lock contention
- Memory buffers are reused across sensor calls rather than reallocated each time



When calling from MATLAB : 

Expects Python code at PMUT/venv

cd /d C:\wherever\they\put\PMUT
python -m venv venv
venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install numpy scipy matplotlib scikit-image pybind11
pip install cupy-cuda12x

To build the CPU module : 
cd TR\TR_Recon_Package_V1
python setup.py build_ext --inplace



## Files needed to run

Download these from `TR/TR_Recon_Package_V1/`:

| File | Purpose |
|---|---|
| `TRHexArray3D_gpu.py` | Main program |
| `coordsHexagonal0p5mm.txt` | Sensor positions |
| `tr_gpu.py` | GPU reconstruction (Python side) |
| `tr_kernel.cu` | GPU reconstruction (CUDA kernel) |
| `tr_cpp/` (folder) | CPU reconstruction source code |
| `setup.py` | Builds the CPU module (`python setup.py build_ext --inplace`) |
| `load_recon_into_slicer.py` | Loads the result into 3D Slicer |
| `slicer_watch.py` | Keeps Slicer open and reloads new results |
| `requirements.txt` | Python packages to install |

Plus at least one data file in `TR/Processed/` (a `.mat` file containing `data_raw`).

Keep the folder structure:

    PMUT/
    └── TR/
        ├── Processed/              <- data (.mat)
        └── TR_Recon_Package_V1/    <- files above


If a new experiment changes:
medium -- change speed of sound
sampling interval dt = 0.016
trigger timing

lines in code must be adjusted