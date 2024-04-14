# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='gravpop_pipe',
    version='0.1.0',
    description='''A helper package to perform data processing and population inference on gravitational wave data''',
    long_description=readme,
    author='Asad Hussain',
    author_email='asadh@utexas.edu',
    url='https://github.com/potatoasad/gravpop_pipe',
    packages=find_packages(exclude=('tests', 'docs', 'dev', 'julia_scripts')),
    entry_points={
        'console_scripts': [
            'gravpop_pipe_run = gravpop_pipe.scripts.gravpop_pipe_run:main',
            'gravpop_pipe_report = gravpop_pipe.scripts.gravpop_pipe_report:main',
        ],
    },
)