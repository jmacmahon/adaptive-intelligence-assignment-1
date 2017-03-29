# Adaptive Intelligence Assignment 1

## Report

The report for this assignment is in `writeup/report.tex`.

## Reproducibility

Code for generating all the figures found in the report is in `figures.py`. It
is designed to be used with IPython, but it can also be used with bare Python on
the terminal.

*Warning* The code will only run on Python 3.

IPython usage:
```
$ ipython

In [1]: %run figures.py

In [2]: performance_surface_graphs()
Out[2]:
{(0, 1): <matplotlib.figure.Figure at 0x7f93483b4f98>,
 (0, 2): <matplotlib.figure.Figure at 0x7f933a02dd68>,
 (1, 2): <matplotlib.figure.Figure at 0x7f9339203908>}

In [3]: plt.show()
```

Terminal usage:
```
$ python -c 'import matplotlib.pyplot as plt; from figures import performance_surface_graphs; performance_surface_graphs(); plt.show()'
```
