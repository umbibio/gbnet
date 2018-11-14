# all .pyx files in a folder
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl


extensions = [
        Extension("cnodes", ["cnodes.pyx"], libraries=['gsl', 'gslcblas']),
        Extension("cchain", ["cchain.pyx"], libraries=['gsl', 'gslcblas']),
        Extension("cbasemodel", ["cbasemodel.pyx"]),
        Extension("cmodels", ["cmodels.pyx"]),
    ]

setup(
    name = 'gbnet',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions, annotate=True)
)

