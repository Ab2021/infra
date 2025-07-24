"""Setup configuration for Building Coverage System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="building-coverage-system",
    version="1.0.0",
    author="Insurance AI Team",
    author_email="ai-team@insurance-company.com",
    description="Advanced building coverage match system with RAG and parallel processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/building-coverage-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "monitoring": [
            "psutil>=5.9.0",
            "memory-profiler>=0.60.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "building-coverage=modules.core.pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
)