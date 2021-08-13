# all .pyx files in a folder
import setuptools
from Cython.Build import cythonize
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
from glob import glob


libs = {
    'libraries': ['gsl', 'gslcblas', 'm'],
    'include_dirs': ['libgbnet/include']
}

args = {
    'extra_compile_args': ['-fopenmp'],
    'extra_link_args': ['-fopenmp']
}

all_lib = [ext for ext in glob('libgbnet/include/*.h')]
all_cpp = [ext for ext in glob('libgbnet/src/*.cpp')]
all_dep = all_lib + all_cpp

extensions = [ Extension("gbnet.ModelORNOR", ["gbnet/ModelORNOR.pyx"]+all_cpp, depends=all_dep, language="c++", **libs, **args)]


with open("README.md", "r") as fh:
    long_description = fh.read()


import re
VERSIONFILE="gbnet/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setuptools.setup(
    name = 'gbnet',
    version=verstr,
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
    entry_points = {
        'console_scripts': ['gbn-ornor-inference=gbnet.commands.ornor_inference:main'],
    },
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions, annotate=False, language_level=3),
    install_requires=['numpy', 'pandas', 'psutil', 'num2words'],
    zip_safe=False,
)
