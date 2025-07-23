#_________________________________________________________________________________________________

# setup.py (this is how you can setup this architecture)
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fast-mctd-mor",
    version="1.0.0",
    author="Fast-MCTD Team",
    author_email="team@fast-mctd.ai",
    description="Fast Monte Carlo Tree Diffusion with Mixture of Recursions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fast-mctd/fast-mctd-mor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
            "deepspeed>=0.9.0",
        ],
        "distributed": [
            "ray[tune]>=2.4.0",
            "optuna>=3.2.0",
        ],
        "jax": [
            "jax>=0.4.10",
            "jaxlib>=0.4.10",
            "flax>=0.7.0",
            "optax>=0.1.5",
            "chex>=0.1.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "fast-mctd=fast_mctd.cli:main",
            "fast-mctd-train=fast_mctd.training.cli:main",
            "fast-mctd-sample=fast_mctd.inference.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fast_mctd": ["configs/*.yaml", "assets/*"],
    },
)

#_______________________________________________________________________________________________________________
