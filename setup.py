from distutils.core import setup
from Cython.Build import cythonize

ext = cythonize("randomforest_functions.pyx")
ext[0].extra_compile_args = ["-O3"]
ext[0].extra_link_args = ["-O3"]

setup(
    name="RandomForestFunctions",
    ext_modules=ext
)
