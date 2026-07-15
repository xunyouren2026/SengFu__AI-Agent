#!/usr/bin/env python3
"""
UFO AGI Unified Framework - Setup Script

Simple setuptools-based installation.
For modern installation, use: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="ufo-agi-framework",
    version="2.0.0",
    description="AGI Unified Framework - Multi-Agent Collaboration Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="UFO Team",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "testing*", "docs*"]),
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.html", "*.css", "*.js"],
    },
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "ufo-agi=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
