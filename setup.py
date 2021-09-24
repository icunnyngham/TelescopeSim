#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Ian Cunnyngham",
    author_email='ian@cunnyngham.net',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Utilize HCIPy to simulate telescope optics, atmosphere for novel distirbuted aperture designs and AO concepts",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='telescope_sim',
    name='telescope_sim',
    packages=find_packages(include=['telescope_sim', 'telescope_sim.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/icunnyngham/TelescopeSim',
    version='1.0.0',
    zip_safe=False,
)
