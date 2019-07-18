import numpy as np
import tensorflow as tf

import interactive_gp
from gpflow.config import default_float, set_default_float

set_default_float(np.float64)
data = tf.cast(tf.reshape(tf.linspace(-5., 10., 100), [-1, 1]), dtype=default_float())
samples_viewer = interactive_gp.SamplesViewer(data, num_samples=10)
panel = samples_viewer.view()
panel.servable()