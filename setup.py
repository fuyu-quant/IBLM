from setuptools import setup, find_packages


setup(
    name='iblm',
    version='0.0.1',
    description='A package of learning models that make predictions from the structure of the model by LLM.',
    packages=find_packages(), 
    install_requires=[
        'langchain==0.0.167',
        'openai==0.27.4',
    ],
    description="",
    author='fuyu-quant',
    url='https://github.com/fuyu-quant/IBLM',
    license='MIT',
)