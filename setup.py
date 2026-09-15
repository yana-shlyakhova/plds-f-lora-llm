from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="plds-f-lora-llm",
    version="0.1.0",
    author="Yana Shlyakhova",
    author_email="yanatext555@gmail.com",
    description="Phase-Locked Dynamic Sampling & Dynamic Fourier-LoRA for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/plds-f-lora-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    keywords="llm, pytorch, lora, fourier-transform, dynamical-systems, rossler-attractor, inference-optimization",
)
