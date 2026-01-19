import os
import glob
import numpy as np

base_dir = os.path.dirname(__file__)

# Compile enforce_connectivity cython code
enforce_connectivity_cython_so_name = glob.glob(os.path.join(base_dir, '_enforce_connectivity*.so'))

if len(enforce_connectivity_cython_so_name) == 0:
    from setuptools import setup
    from Cython.Build import cythonize

    setup(
        ext_modules=cythonize(os.path.join(base_dir, '_enforce_connectivity_mask.pyx')),
        include_dirs=[np.get_include()],
        script_args=["build_ext", "--inplace"],
    )
