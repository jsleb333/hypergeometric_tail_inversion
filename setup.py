import setuptools
from version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hypergeo",
    version=__version__,
    author="Jean-Samuel Leboeuf",
    author_email="jean-samuel.leboeuf.1@ulaval.ca",
    description="An implementation of the hypergeometric tail pseudo-inverse used in the paper Improving generalization bounds for VC classes using the hypergeometric tail inversion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jsleb333/hypergeometric_tail_inversion",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'python2latex==0.4.1',
        'colorama',
        'graal_utils @ git+https://github.com/GRAAL-Research/graal-utils.git',
        'pandas',
        'xarray',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    data_files=[('', ['version.py'])]
)
