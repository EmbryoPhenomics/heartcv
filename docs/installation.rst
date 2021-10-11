.. _installation:

Installation
============

Required dependencies
---------------------

- Vuba_
- SciPy_
- NumPy_
- Numba_
- matplotlib_
- more_itertools_
- OpenCV_

.. _Vuba: https://github.com/EmbryoPhenomics/vuba
.. _SciPy: https://github.com/scipy/scipy
.. _NumPy: https://github.com/numpy/numpy
.. _Numba: https://github.com/numba/numba
.. _matplotlib: https://github.com/matplotlib/matplotlib
.. _more_itertools: https://github.com/more-itertools/more-itertools
.. _Opencv: https://github.com/opencv/opencv

Optional dependencies for the validation interface
--------------------------------------------------

- dash_ (1.9.1)
- dash_table_ (4.6.1)
- plotly_
- flask_ 
- pandas_
- dataclasses_

.. _dash: https://github.com/plotly/dash
.. _dash_table: https://github.com/plotly/dash-table
.. _plotly: https://github.com/plotly/plotly.py
.. _flask: https://github.com/pallets/flask/
.. _pandas: https://github.com/pandas-dev/pandas
.. _dataclasses: https://github.com/ericvsmith/dataclasses

Instructions
------------

HeartCV is a pure Python package and can be installed from PyPi_ using ``pip``::

    $ pip install heartcv

Note that OpenCV is not declared as a required dependency due to the variety of ways one may need to go about installing it on different platforms, and so you will need to install separately if don't already have it. 

If you would like to install the latest development version, you will need to clone the repository on GitHub and install it as follows::

    $ git clone https://github.com/EmbryoPhenomics/heartcv.git
    $ cd heartcv
    $ pip install .

.. _Pypi: https://pypi.org/