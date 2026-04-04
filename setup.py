"""Makes the project installable as a package (enables `from src.x import y`)."""
from setuptools import setup, find_packages

setup(
    name="mindforge",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
