# all .pyx files in a folder
import setuptools
from Cython.Build import cythonize
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy
import cython_gsl


libs = {
    'libraries': cython_gsl.get_libraries(),
    'library_dirs': [cython_gsl.get_library_dir()],
    'include_dirs': [
        cython_gsl.get_cython_include_dir(),
        numpy.get_include(),
    ]
}

extensions = [
        Extension("gbnet.cchain", ["gbnet/cchain.pyx"], **libs),
        Extension("gbnet.cnodes", ["gbnet/cnodes.pyx"], **libs),
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'gbnet',
    version='0.1.dev1',
    author="Argenis Arriojas",
    author_email="arriojasmaldonado001@umb.edu",
    description=(
        'A Bayesian Networks approach for infering active Transcription Factors '
        'using logic models of transcriptional regulation'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/umbibio/gbnet',
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],

    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions, annotate=False),
    zip_safe=False,
)
