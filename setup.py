from setuptools import setup, find_packages

setup(
    name="holographic-qc",
    version="0.1.0",
    author="Ahmed Ali",
    author_email="ahmed19999520@gmail.com",
    description="Holographic quantum computing library based on AdS3/CFT2",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ahmed19999520-alt/holographic-qc",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "sympy>=1.12",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "tqdm>=4.65",
        "h5py>=3.8",
    ],
    extras_require={
        "tf": ["tensorflow>=2.12"],
        "torch": ["torch>=2.0", "torchvision>=0.15"],
        "qiskit": ["qiskit>=0.44", "qiskit-aer>=0.12"],
        "all": [
            "tensorflow>=2.12",
            "torch>=2.0",
            "qiskit>=0.44",
            "qiskit-aer>=0.12",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)