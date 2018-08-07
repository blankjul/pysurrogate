from setuptools import setup, find_packages

setup(
    name="pysurrogate",
    version="0.1.0",
    author="Julian Blank",
    description=("Surrogate Models which can be used for optimization or other tasks."),
    license='MIT',
    keywords="optimization",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn']
)
