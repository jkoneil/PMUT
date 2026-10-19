"""CPU fallback build (C++/OpenMP)
Build:  python3 setup.py build_ext --inplace
(The GPU path does NOT need this; it only needs CuPy.)"""

import sys
from setuptools import setup, Extension
import pybind11

include_dirs = [pybind11.get_include()]
if sys.platform == "darwin":
    include_dirs.append("/opt/homebrew/opt/libomp/include")
    compile_args = ["-O3", "-std=c++17", "-Xpreprocessor", "-fopenmp"]
    link_args = ["-L/opt/homebrew/opt/libomp/lib", "-lomp"]
elif sys.platform.startswith("win"):
    compile_args = ["/O2", "/std:c++17", "/openmp"]
    link_args = []
else:  # Linux
    compile_args = ["-O3", "-std=c++17", "-fopenmp"]
    link_args = ["-fopenmp"]
 
ext = Extension(
    "tr_cpp",
    sources=["tr_cpp/tr_reconstruction.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    language="c++",
)
setup(name="tr_cpp", ext_modules=[ext])
