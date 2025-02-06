import setuptools

with open('README.md', 'r') as ff:
    long_description = ff.read()

setuptools.setup(
    name='qiskit-torch-module',
    version='0.1.0b',
    description='Fast quantum neural networks with automatic differentiation. '
                'This is a modified sequential version which allows inputting circuits during runtime.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Nico Meyer',
    author_email='nico.meyer@iis.fraunhofer.de',
    license='Apache License 2.0',
    platforms='any',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.10.13',
    install_requires=[
        'qiskit>=0.45.1',
        'torch~=2.2.0',
        'qiskit-algorithms>=0.2.1'
    ],
    include_package_data=True
)
