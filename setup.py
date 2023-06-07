from setuptools import setup, find_packages


setup(
    name='inductivebiaslearning',
    version='0.0.0',
    packages=find_packages(),  # ここでパッケージ内のモジュールが自動的に見つけられます
    install_requires=[  # 依存関係リスト
        'langchain==0.0.167',
        'openai==0.27.4',
    ],
    description="",
    author='fuyu-quant',  
    license='MIT'
)