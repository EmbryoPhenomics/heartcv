.. _installation:

Installation
============

Required dependencies
---------------------

- Vuba_
- SciPy_
- NumPy_
- OpenCV_ (opencv-python)
- tqdm_

.. _Vuba: https://github.com/EmbryoPhenomics/vuba
.. _SciPy: https://github.com/scipy/scipy
.. _NumPy: https://github.com/numpy/numpy
.. _OpenCV: https://github.com/opencv/opencv
.. _tqdm: https://github.com/tqdm/tqdm

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

Note that for some Linux distributions such as Ubuntu you may need to install OpenCV using ``apt`` prior to installing HeartCV. Using the above command will install all of the required dependencies by default. 

If you would like to install the latest development version, you will need to clone the repository on GitHub and install it as follows::

    $ git clone https://github.com/EmbryoPhenomics/heartcv.git
    $ cd heartcv
    $ pip install .

Like above, this will install all the required dependencies for vuba.

.. _Pypi: https://pypi.org/