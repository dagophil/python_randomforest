from numpy.distutils.core import setup
from Cython.Build import cythonize
import sys

ext = cythonize("randomforest/randomforest_functions.pyx")

if not sys.platform.startswith("win"):
    ext[0].extra_compile_args = ["-O3"]
    ext[0].extra_link_args = ["-O3"]

setup(
    name="randomforest",
    packages=["randomforest"],
    ext_modules=ext
)
