from setuptools import setup, find_packages

setup(
    name='IHSetJara',
    version='1.1.4',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'numba',
        'fast_optimization @ git+https://github.com/defreitasL/fast_optimization.git'
    ],
    author='Changbin Lim',
    author_email='@unican.es',
    description='IH-SET Jara et al. (2022)',
    url='https://github.com/IHCantabria/IHSetJara',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)