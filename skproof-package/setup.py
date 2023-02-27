from setuptools import setup

setup(
    name='skproof',
    version='0.1.0',    
    description='SciKit learn compatible library for generating ZK proofs of execution',
    url='https://github.com/0x3327/skproof.git',
    author='Aleksandar Veljković, 3327.io',
    author_email='aleksandar.veljkovic@mvpworkshop.co',
    license='BSD 2-clause',
    packages=['skproof', 'skproof.float_num', 'skproof.mlp'],
    install_requires=['numpy','sklearn'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.10',
    ],
)