from setuptools import setup, Extension
import pybind11

ext = Extension(
    "tr_cpp",
    sources=["tr_cpp/tr_reconstruction.cpp"],
    include_dirs=[
        pybind11.get_include(),
        "/opt/homebrew/opt/libomp/include"
    ],
    extra_compile_args=["-O3", "-std=c++17", "-Xpreprocessor", "-fopenmp"],
    extra_link_args=["-L/opt/homebrew/opt/libomp/lib", "-lomp"],
    language="c++",
)

setup(name="tr_cpp", ext_modules=[ext])

''' To run : python3 setup.py build_ext --inplace '''