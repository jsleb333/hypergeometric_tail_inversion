# Improving Generalization Bounds for VC Classes Using the Hypergeometric Tail Inversion

## Preface
This directory provides an implementation of the algorithms used to compute the hypergeometric tail pseudo-inverse, as well as the code used to produce all figures of the paper "Improving Generalization Bounds for VC Classes Using the Hypergeometric Tail Inversion" by Leboeuf, LeBlanc and Marchand.

## Installation

To run the scripts, one must first install the package and its requirements.
To do so, run the following command from the root directory:
```
pip install .
```

## Requirements

The code was written to run on Python 3.8 or more recent version.
The requirements are shown in the file `requirements.txt` and can be installed using the command:
```
pip install -r requirements.txt
```

## The code

The code is split into 2 parts: the 'hypergeo' package and the 'scripts' directory.

The hypergeo package implements the utilities regarding the hypergeometric distribution (to compute the tail and its inverse), the binomial distribution (reimplementing the inverse as the scipy version suffered from numerical unstabilities) and some generalization bounds.

The scripts files produce the figures found in the paper using the hypergeo package.
All figures are generated directly in LaTeX using the package `python2latex`.
To run a script, navigate from the command line to the directory root directory of the project and run the command
```
python "./scripts/<dir_name>/<file_name>.py"
```

The code does not provide command line control on the parameters of each script.
However, each script is fairly simple, and parameters can be directly changed in the `__main__` part of the script.

### Scripts used in the body of the paper

- Section 3.3: The ghost sample trade-off.
In this section, we claim that optimizing m' gives relative gain between 8% and 10%. To obtain these number, you need to run the file `mprime_tradeoff/generate_mprime_data.py` to first generate the data, and then run `mprime_tradeoff/stats.py`.

- Section 5: Numerical comparison.
Figure 1a and 1b are obtain by executing the scripts `bounds_comparison/bounds_comparison_risk.py` and `bounds_comparison/bounds_comparison_d.py` respectively.
Figure 2a and 2b are obtain by executing the scripts `bounds_comparison/bounds_comparison_m.py`, the first setting the variable `risk` to 0, the second by setting it equal to 0.1.


### Scripts used in the appendices of the paper

- Appendix B: Overview of the hypergeometric distribution.
Figure 3 is generated from `hypergeometric_tail/hyp_tail_plot.py`.
Figure 4 is generated from `hypergeometric_tail/hyp_tail_inv_plot.py`.
Algorithm 1 is implemented in the hypergeo file `hypergeo/hypergeometric_distribution.py` as the function `hypergeometric_tail_inverse`.
Algorithm 2 is implemented in the hypergeo file `hypergeo/hypergeometric_distribution.py` as the function `berkopec_hypergeometric_tail_inverse`.

- Appendix D: In-depth analysis of the ghost sample trade-off.
Figure 5 is generated from `mprime_tradeoff/plot_epsilon_comp.py`.
Figure 6 is generated from `mprime_tradeoff/plot_mprime_best.py`.

- Appendix E: The hypergeometric tail inversion relative deviation bound.
To generate Figure 7 and 8, you must first run the file `relative_deviation_mprime_tradeoff/mprime_tradeoff_relative_deviation.py` to generate the data, then run the script `relative_deviation_mprime_tradeoff/plot_epsilon_comp.py` to produce Figure 7 and `relative_deviation_comparison/plot_mprime_best.py` to produce Figure 8.

- Appendix G: The hypergeometric tail lower bound .
Figure 9 is generated from `lower_bound/lower_bound_comparison_risk.py`.

- Appendix F: Further numerical comparisons.
Figure 10 and 12a are generated from `bounds_comparison/bounds_comparison_risk.py` by changing the parameters of the scripts.
Figure 11 and 12b is generated from `bounds_comparison/bounds_comparison_m.py` by changing the parameters of the scripts.
Figure 13a and 13b are generated from `bounds_comparison/sample_compression_comparison_risk.py` and `bounds_comparison/sample_compression_comparison_m.py` respectively.

### Other
The script `pseudo-inverse_benchmarking/pseudo-inverse_benchmarking.py` benchmarks the various algorithms used to invert the hypergeometric tail.
The 'tests' directory contains unit tests using the package `pytest`.
