from setuptools import setup
from Cython.Build import cythonize
import os

def build_cython_files(path):
	cython_files = []
	for root,_,files in os.walk(path):
		for file in files:
			if file.endswith('.pyx'):
				filename = os.path.join(root, file)
				cython_files.append(filename)

	setup(
		ext_modules=cythonize(cython_files)
	)





