from distutils.core import setup
from Cython.Build import cythonize

ext = cythonize("gini.pyx")
ext[0].extra_compile_args = ["-O3"]
ext[0].extra_link_args = ["-O3"]

setup(
    name="GiniScorer",
    ext_modules=ext
)
