# all .pyx files in a folder
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy
import cython_gsl


extensions = [
        Extension("cnodes", ["cnodes.pyx"], libraries=['gsl', 'gslcblas'], include_dirs=[numpy.get_include()]),
        Extension("cchain", ["cchain.pyx"], libraries=['gsl', 'gslcblas'], include_dirs=[numpy.get_include()]),
        # Extension("cbasemodel", ["cbasemodel.pyx"]),
        # Extension("cmodels", ["cmodels.pyx"]),
    ]

setup(
    name = 'gbnet',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize( extensions, annotate=True, include_path=[ '../', './' ])
)

