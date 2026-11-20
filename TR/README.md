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
Detection past ~13mm or closer than ~1mm 
Might need to adjust hard-coded values, dt and co, zmax, BLANK_MM


If you manually open 3D slicer before running the program, you need to go to the python console in slicer and open the slieer_watch file : 
p = "/path/slicer_watch.py"
exec(open(p).read(), {"__file__": p})





MATLAB : 

## Data processing pipeline

Raw Verasonics data goes through three steps, all run by `process_dataset.m`:

| Step | Function | Input | Output (in `Processed/`) |
|---|---|---|---|
| 1. Average | `PA_average_dataset.m` | Raw dataset folder (frames + time tags) | `<name>_<N>_Averaged.mat` — frames averaged per MUX (`data_avg`) |
| 2. Map & merge | `PA_map_and_merge.m` | `<name>_<N>_Averaged.mat` | `<name>_PMUT_61_channels_64.mat` — 61-channel data (`data_raw`, `data_filt`, `*_interp`) |
| 3. Reconstruct | `TR/TR_Recon_Package_V1/TRHexArray3D_gpu.py` | `<name>_PMUT_61_channels_64.mat` | 3D image, shown in 3D Slicer |

### Running it

1. Open `process_dataset.m` in MATLAB and click **Run**.
2. In the folder browser, select one **dataset folder** (e.g. `pmut_#46_hair_1`).
3. Wait — progress is printed for each step, and the result opens in 3D Slicer.

The dataset name is taken from the folder name chosen at acquisition; nothing needs to be edited per dataset.

### Folder structure

**Raw data** (from the Verasonics, one folder per day, one subfolder per dataset):

    Data_09252026/
    └── pmut_#46_hair_1/                  <- select this folder
        ├── pmut_#46_hair_1_timetag/        Verasonics time tags (Timetag_1.mat, ...)
        ├── NI_timetag/                     NI card time tags (time_tag_pmut_#46_hair_1_1.mat, ...)
        └── pmut_#46_hair_1/                raw frames (Frame_1.mat, Frame_2.mat, ...)

Dataset naming: `pmut_#<device>_<target>_<n>`, e.g. `pmut_#46_hair_1` = device #46, hair target, first dataset.

**Processing folder** (code and outputs, kept together):

    <processing folder>/
    ├── process_dataset.m             run this
    ├── PA_average_dataset.m          step 1
    ├── PA_map_and_merge.m            step 2
    ├── Processed/                    outputs of steps 1 and 2 (created automatically)
    ├── TR/TR_Recon_Package_V1/       step 3: Python reconstruction
    └── venv/                         Python environment (see Setup)

The raw data can stay anywhere (local drive or NAS); it is only read, never modified.

### Settings (top of `process_dataset.m`)

| Setting | Default | Meaning |
|---|---|---|
| `num_avg` | 400 | Frames averaged per MUX |
| `openSlicer` | true | Show the result in 3D Slicer |
| `pyOptions` | `''` | Extra options for the reconstruction (e.g. `'--zmax 8'`) |

Hardware constants (Verasonics channels `[1, 61, 68, 128, 129, 189, 196, 256]`, 8 NI time-tag files, and the MUX-to-pMUT lookup table for the new die batch) are inside the functions and only change if the wiring or pMUT batch changes.

### Notes

- `data_filt` (11–16 MHz band-pass) is saved for reference; the reconstruction uses `data_raw` with its own 12 MHz low-pass filter.
- `.mat` files are saved with `-v7` so Python can read them.
- The original scripts (`YTang_PA_Data_Quick_AveragingV3.m`, `Map_and_Merge_V6.m`, `TRHexArray3D1mmpMUT_TK.m`) are unchanged and can still be used manually.