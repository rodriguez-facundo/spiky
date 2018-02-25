from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spiky',

    version='1.0.3',

    description='Spike sorting based on Gaussian Mixture Model',

    long_description=long_description,

    url='https://github.com/rodriguez-facundo/spiky',

    author='F. Rodriguez',

    author_email='frodriguez4600@gmail.com',

    classifiers=[ 'Development Status :: 4 - Beta',
                    'Intended Audience :: Science/Research',
                    'Topic :: Scientific/Engineering :: Medical Science Apps.',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3.6',
            ],

    keywords='spike sorting',

    packages=['spiky'],

    install_requires=[  'numpy',
                        'matplotlib',
                        'progressbar2',
                        'pywavelets',
                        'scipy',
                        'sklearn'
                    ],

    extras_require={},

    package_data={},

    data_files=[],

    entry_points={},
)
