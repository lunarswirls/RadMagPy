# Imports:
from setuptools import setup, find_packages

setup(
    name="radmagpy",
    # Version scheme is last date updated in YYYY.MM.DDv format
    # where 'v' may increment [a...z] for multiple releases on the same day
    version="2025.11.06a",
    description="Radar and Magnetism (RadMag) Python Package",
    author="Dany Waller",
    author_email="dany.c.waller@gmail.com",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11, <4",
)