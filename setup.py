from setuptools import setup, find_packages

setup(
    name="autotau",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
        "pandas>=1.0.0",
        "tqdm>=4.45.0",
    ],
    author="Donghao Li",
    author_email="lidonghao100@outlook.com",
    description="自动化时间常数tau拟合工具，支持并行处理",
    keywords="tau, fitting, exponential, signal processing, parallel",
    url="https://github.com/Durian-Leader/autotau",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
)
