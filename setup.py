from setuptools import setup, find_packages


setup(
    name='iblm',
    version='0.0.20',
    description='Inductive Bias Learning Models',
    packages=find_packages(), 
    install_requires=[
        'langchain==0.0.167',
        'openai==0.27.4',
    ],
    author='fuyu-quant',
    url='https://github.com/fuyu-quant/IBLM',
    license='MIT',
)