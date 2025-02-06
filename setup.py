from setuptools import find_packages, setup

setup(
    name='compilation_qwgan',
    version='0.1.0',
    author='Marvin Richter, Abhishek Dubey',
    author_email='marvin.richter@chalmers.se, abhishek.yogendra.dubey@iis.fraunhofer.de (aabhishekdubey479@gmail.com)' ,
    license='Apache License 2.0',
    python_requires='>=3.12',
    install_requires = [
        'qiskit>=1.0.0',
        'qiskit_algorithms>=0.2.1',
        ],
    packages=find_packages(),
)