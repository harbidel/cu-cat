from setuptools import setup

# if __name__ == "__main__":
setup(
    name='cu-cat',
    version='v0.07.00',
    # cmdclass=versioneer.get_cmdclass(),
    # packages = find_packages(),
    platforms='any',
    description = 'An end-to-end gpu Python library that encodes categorical variables into machine-learnable numerics',
    long_description=open("./README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/graphistry/cu-cat',
    download_url= 'https://github.com/graphistry/cu-cat',
    python_requires='>=3.7',
    author='The Graphistry Team',
    author_email='pygraphistry@graphistry.com',
    # install_requires=core_requires,
    # extras_require=extras_require,
    license='BSD',
    
    keywords=['cudf', 'cuml', 'GPU', 'Rapids']
)

