from setuptools import setup, find_packages

setup(
    name="sero",
    version="0.1.0",
    description="SERO: Self-Evolving Role Orchestration for Multi-Agent Systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["scripts", "results", "Benchmark"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "openai>=1.0.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
