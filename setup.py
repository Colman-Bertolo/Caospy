import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# Requerimientos

REQUIREMENTS = ["numpy", "pandas", "scipy", "sympy", "matplotlib"]

# Version
with open(PATH / "caospy" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


# Descripción del proyecto
with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()

# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="Caospy",
    version=VERSION,
    description="Dynamic systems analysis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="The MIT License",
    author="Juan Colman, Sebastián Bertolo",
    author_email="juancolmanot@gmail.com",
    url="https://github.com/Colman-Bertolo/Caospy",
    packages=["caospy"],
    install_requires=REQUIREMENTS,
    keywords=["caospy", "dynamic system", "chaos"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
)
