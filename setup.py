from setuptools import setup

setup(
    name='posembd',
    version='0.0.1',
    description='POS Embeddings',
    author='Paulo Augusto de Lima Medeiros',
    author_email='pauloaugusto99@ufrn.edu.br',
    license='unlicense',
    packages=['posembd', 'posembd.models', 'posembd.io', 'posembd.datasets', 'posembd.base'],
    zip_safe=False
)
