from setuptools import find_packages, setup
from bbox_to_seg.__init__ import __version__

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="bbox_to_seg",
    packages=find_packages(),
    version=__version__,
    url='https://bitbucket.org/username/package_name/src/master/',
    description="A tool to convert bbox datasets into segmentation datasets.",
    author="Enrique Fernández-Laguilhoat Sánchez-Biezma",
    license="TO_BE_FILLED_IN",
    install_requires=requirements,
)