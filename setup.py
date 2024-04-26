import pathlib
import setuptools
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="torch_choice",
    version="1.0.6",
    description="A Pytorch Backend Library for Choice Modelling",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://gsbdbi.github.io/torch-choice/",
    author="Tianyu Du, Ayush Kanodia, and Susan Athey",
    author_email="tianyudu@stanford.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    # packages=["torch_choice"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    # install_requires=["torch"],
    install_requires=[
        "numpy>=1.22",
        "termcolor>=1.1.0",
        "scikit-learn",
        "pandas>=1.4.3",
        "tabulate>=0.8.10",
        "torch>=1.12.0",
        "pytorch-lightning>=1.6.3",
    ]
    # entry_points={
    #     "console_scripts": [
    #         "realpython=.__main__:main",
    #     ]
    # },
)
