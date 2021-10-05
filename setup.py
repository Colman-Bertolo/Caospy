

from setuptools import setup 

# Descripción del proyecto 
with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()

# Requerimientos 

REQUIREMENTS = ["numpy", "pandas", "scipy", "sympy", "matplotlib"] 

# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="Caospy",
    version="0.1.1",
    description="Dynamic systems analysis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Juan Colman, Sebastián Bertolo",
    author_email="juancolmanot@gmail.com",
    url="https://github.com/Colman-Bertolo-DiSCSI2021/Caospy",
    packages=[
        "caospy",
    ],
    install_requires=REQUIREMENTS,
    keywords=["caospy", "dynamic system", "chaos"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
   # include_package_data=True,
)

