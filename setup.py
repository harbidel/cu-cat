#!/usr/bin/env python

from setuptools import setup, find_packages
#FIXME: prevents pyproject.toml - same as https://github.com/SciTools/cartopy/issues/1270
import versioneer

def unique_flatten_dict(d):
  return list(set(sum( d.values(), [] )))

core_requires = [
  'numpy',
  'pandas',
  'setuptools',
  'logging',
  'typing',
  'sklearn',
  'cuml',
  'cudf',
  'cupy'
]

from setuptools import setup

# if __name__ == "__main__":
setup(
    name='cu-cat',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages = find_packages(),
    platforms='any',
    description = 'An end-to-end gpu Python library that encodes categorical variables into machine-learnable numerics',
    long_description=open("./README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/graphistry/cu-cat',
    download_url= 'https://github.com/graphistry/cu-cat',
    python_requires='>=3.9',
    author='The Graphistry Team',
    author_email='pygraphistry@graphistry.com',
    install_requires=core_requires,
    license='BSD',
    dependency_links='https://pypi.nvidia.com',
    keywords=['cudf', 'cuml', 'GPU', 'Rapids']
)

