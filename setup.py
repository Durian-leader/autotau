from setuptools import setup, find_packages
import pathlib

# 读取README.md文件内容
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="autotau",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
        "pandas>=1.0.0",
        "tqdm>=4.45.0",
    ],
    extras_require={
        "accelerate": ["numba>=0.50.0"],  # Numba JIT 编译加速（可选，5-10x 加速）
    },
    author="Donghao Li",
    author_email="lidonghao100@outlook.com",
    description="Automated tau fitting with flexible parallelization and global optimization (200-1500x speedup)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="tau, fitting, exponential, signal processing, parallel, optimization, numba",
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
    package_data={"autotau": ["docs/*.md"]},
    include_package_data=True,
)