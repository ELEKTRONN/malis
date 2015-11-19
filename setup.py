#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

#ext_modules = [Extension("_malis", ["_malis.pyx", "_malis_lib.cpp"], language='c++',)]

#setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)


############################################
import os
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
 return open(os.path.join(os.path.dirname(__file__), fname)).read()



pysmall = Extension('pysmall',
    sources = ['pysmall.pyx', 'small.cpp'],
    include_dirs = ['include/'])



ext_modules = [Extension("malis._malis", ["malis/_malis.pyx", "malis/_malis_lib.cpp"])]

setup(
    name = "malis",
    version = "0.1",
    packages = find_packages(), 
    ext_modules=ext_modules,
    cmdclass = {'build_ext': build_ext}, 
    install_requires = ['cython>=0.21.1',], 
    author = "Srini Turaga, Marius Killinger",
    description = ("Structured loss function for supervised learning of segmentation and clustering"),
    long_description = read('README.rst'),
    license = "GPL",
    keywords = "",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Scientific/Engineering :: Information Analysis",],
)
