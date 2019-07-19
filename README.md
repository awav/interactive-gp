Interactive Gaussian Processes
===
The collection of GP visualizations. The GP computation part benefits from TensorFlow 2.0 and [GPflow 2.0](https://github.com/GPflow/GPflow/tree/awav/gpflow-2.0), whereas the visualization implemetation sits on top of the [holoviews](http://holoviews.org/) framework, which in turn based on [bokeh](https://bokeh.pydata.org/en/latest/).

_Any contributions or ideas about visualizations which you think could be helpful are welcome._


### Implemented:

* GP samples for squared exponential, matern52, matern32, matern12, arccosine 0th order and linear kernels

### Plans to add:

* GPR model visualization with different kernels and ability to add new data points
* SVGP online training, moving inducing points

## Install package

Using conda is optional, but still it is recommended in the case when you don't want to collide with existing packages like TensorFlow or GPflow

```bash
conda activate your-env
(your-env) pip install -r requirements.txt
(your-env) pip install -e .
```

## How to run interactive HTML page:

```base
(your-env) bokeh serve --show apps/samples.py
```

You can run the same plots in a notebook - check `notebooks` folder.
