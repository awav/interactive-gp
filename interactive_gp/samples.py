from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, Union

import holoviews as hv
import numpy as np
import panel as pn
import tensorflow as tf
from holoviews import opts, streams

import gpflow
from gpflow.config import default_float, default_jitter
from gpflow.utilities.ops import eye

pn.extension('katex')
hv.extension('bokeh')

blue = "#1f77b4"
orange1 = "#FF9636"
orange2 = "#FF5C4D"


__all__ = [
    "SamplesViewer"
]


def assign_parameters(*updates):
    for p, v in updates:
        if isinstance(p, (tf.Variable, gpflow.Parameter)):
            p.assign(tf.cast(v, p.dtype))
        elif callable(p):
            p(v)
        else:
            assert False, "Input parameter type is not recognisable"


class OptimizedKernel:
    def __init__(self, kernel: gpflow.kernels.Kernel):
        self.kernel = kernel

        @tf.function
        def tf_kernel_call(data: tf.Tensor) -> tf.Tensor:
            return kernel(data)

        @tf.function
        def tf_sample_call(cov: tf.Tensor, epsilon: tf.Tensor) -> tf.Tensor:
            cov += eye(cov.shape[0], default_jitter(), dtype=default_float())
            cov_cholesky = tf.linalg.cholesky(cov)
            return cov_cholesky @ epsilon

        self.tf_sample_call = tf_sample_call
        self.tf_kernel_call = tf_kernel_call

    def __hash__(self):
        name = type(self.kernel).__name__
        values = [v.numpy() for v in self.kernel.variables]
        h = (name, *values)
        return hash(h)


@dataclass
class Kernel(tf.keras.models.Model):
    channels: List[streams.Pipe]
    kernel: gpflow.kernels.Kernel = field(init=False)
    widgets: List[pn.widgets.Widget] = field(init=False)

    def setup_kernel(self):
        kernel_cls = getattr(gpflow.kernels, type(self).__name__)
        kernel = kernel_cls()
        self.kernel = kernel
        self.tf_kernel = OptimizedKernel(self.kernel)

    def make_update_parameter_cb(self, parameter: Union[gpflow.Parameter, Callable]):
        def update_parameter(event):
            if event.new is not None:
                assign_parameters((parameter, event.new))
                channels = self.channels if isinstance(self.channels, list) else [self.channels]
                for channel in channels:
                    channel.send(self.tf_kernel)

        return update_parameter

    def __str__(self):
        return type(self).__name__


@dataclass
class SquaredExponential(Kernel):
    kernel: gpflow.kernels.Kernel = field(init=False)
    equation: pn.pane.LaTeX.__class__ = field(
        init=False, default=pn.pane.LaTeX(r"$k(x, y) = \sigma^2 \exp(-\frac{||x - y||^2}{2 \theta^2})$", margin=20))

    def __post_init__(self):
        self.setup_kernel()

        variance = pn.widgets.FloatSlider(name='Variance', start=1e-6, end=5, value=1.0, step=1e-2)
        lengthscale = pn.widgets.FloatSlider(name="Lengthscale", start=1e-6, end=5, value=1.0, step=1e-2)

        cb = self.make_update_parameter_cb
        variance.param.watch(cb(self.kernel.variance), "value")
        lengthscale.param.watch(cb(self.kernel.lengthscale), "value")

        self.widgets = [self.equation, variance, lengthscale]


@dataclass
class Matern52(SquaredExponential):
    kernel: gpflow.kernels.Kernel = field(init=False)
    equation: pn.pane.LaTeX.__class__ = field(
        init=False,
        default=pn.pane.LaTeX(
            r"""$k(x, y) = \sigma^2 (1 + \frac{\sqrt{5} |x-y|}{\theta} + \frac{5}{3}\frac{|x-y|^2}{\theta^2})
        \exp(-\sqrt{5}\frac{|x - y|}{\theta})$""",
            margin=20))


@dataclass
class Matern32(SquaredExponential):
    kernel: gpflow.kernels.Kernel = field(init=False)
    equation: pn.pane.LaTeX.__class__ = field(
        init=False,
        default=pn.pane.LaTeX(
            r"$k(x, y) = \sigma^2 (1 + \sqrt{3}\frac{|x-y|}{\theta})\exp(-\frac{|x - y|}{\theta})$", margin=20))


@dataclass
class Matern12(SquaredExponential):
    kernel: gpflow.kernels.Kernel = field(init=False)
    equation: pn.pane.LaTeX.__class__ = field(
        init=False, default=pn.pane.LaTeX(
            r"$k(x, y) = \sigma^2 \exp(-\frac{|x - y|}{\theta})$", margin=20))


@dataclass
class ArcCosine(Kernel):
    kernel: gpflow.kernels.Kernel = field(init=False)

    def __post_init__(self):
        self.setup_kernel()

        #         def assign_order(new_value):
        #             self.kernel.order = int(new_value)

        #         self.kernel.order = gpflow.Parameter(0, dtype=gpflow.config.default_int())

        #         order = pn.widgets.Select(name="Order", value=0, options=[0, 1, 2])
        variance = pn.widgets.FloatSlider(name="Variance", value=1.0, start=1.0, end=3.0, step=1e-2)
        weight_variances = pn.widgets.FloatSlider(name="Weight variances", value=1.0, start=1e-6, end=3.0, step=1e-2)
        bias_variance = pn.widgets.FloatSlider(name="Bias variance", value=1.0, start=1e-6, end=3.0, step=1e-2)

        cb = self.make_update_parameter_cb
        #         order.param.watch(cb(self.kernel.order), "value")
        variance.param.watch(cb(self.kernel.variance), "value")
        weight_variances.param.watch(cb(self.kernel.weight_variances), "value")
        bias_variance.param.watch(cb(self.kernel.bias_variance), "value")

        #         self.widgets = [order, variance, weight_variances, bias_variance]
        self.widgets = [variance, weight_variances, bias_variance]


@dataclass
class Linear(Kernel):
    kernel: gpflow.kernels.Kernel = field(init=False)
    equation: pn.pane.LaTeX.__class__ = field(
        init=False, default=pn.pane.LaTeX(r"$k(x, y) = \sigma^2 x y$", margin=20))

    def __post_init__(self):
        self.setup_kernel()
        variance = pn.widgets.FloatSlider(name="Variance", value=1.0, start=1e-6, end=2.0, step=1e-2)
        variance.param.watch(self.make_update_parameter_cb(self.kernel.variance), "value")
        self.widgets = [self.equation, variance]


class SamplesPipe(streams.Pipe):
    def __init__(self, input_data: tf.Tensor, epsilon: tf.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = input_data
        self.epsilon = epsilon

    def transform(self):
        kernel = self.contents["data"]
        return dict(data=self.get_data(kernel))

    @lru_cache(maxsize=256)
    def get_data(self, kernel: gpflow.kernels.Kernel):
        epsilon = self.epsilon
        xs = self.input_data
        cov = kernel.tf_kernel_call(xs)
        ys = kernel.tf_sample_call(cov, epsilon)
        xs = tf.reshape(xs, (-1, ))
        return (xs.numpy(), ys.numpy(), cov.numpy())


@dataclass
class KernelsController:
    channels: List[streams.Pipe]
    widget_box: pn.WidgetBox = pn.WidgetBox()
    selector_widget_kernel: pn.widgets.Widget = pn.widgets.Select()
    widget_kernel_instances: List[Kernel] = field(init=False, default=None)
    widget_kernel_classes: List[Type] = field(
        default_factory=lambda: [SquaredExponential, Matern52, Matern32, Matern12, ArcCosine, Linear])

    def __post_init__(self):
        self.widget_kernel_instances = [cls(self.channels) for cls in self.widget_kernel_classes]

        first_instance = self.widget_kernel_instances[0]
        self.selector_widget_kernel.options = self.widget_kernel_instances
        self.selector_widget_kernel.value = first_instance

        def update_parameter_box(event):
            kernel_widget = first_instance if event.new is None else event.new
            self.widget_box.objects = kernel_widget.widgets
            for channel in self.channels:
                channel.send(kernel_widget.tf_kernel)

        self.selector_widget_kernel.param.watch(update_parameter_box, "value")
        self.selector_widget_kernel.param.trigger("value")

    def view(self):
        return pn.Column(self.selector_widget_kernel, self.widget_box)


@dataclass
class SamplesViewer:
    data: tf.Tensor
    num_samples: int = 3
    epsilon: tf.Tensor = field(init=False, default=None)
    kernels_controller: KernelsController = field(init=False, default=None)
    maps: Mapping[str, hv.DynamicMap] = field(init=False, default=None)
    streams: Mapping[str, Any] = field(init=False, default=None)

    vlines_positions: Tuple[float, float] = (0.0, 1.0)
    samples_data: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        num_grid_pts = self.data.shape[0]
        self.epsilon = tf.random.normal((num_grid_pts, self.num_samples), dtype=default_float())
        samples_pipe = SamplesPipe(self.data, self.epsilon)
        self.kernels_controller = KernelsController([samples_pipe])
        samples_map = hv.DynamicMap(self.update_samples, streams=[samples_pipe])

        control_vline_stream = hv.streams.Draw(rename=dict(x="x1", y="x2"))
        control_vline_map = hv.DynamicMap(self.update_control_vlines, streams=[control_vline_stream, samples_pipe])
        vlines_map = hv.DynamicMap(self.update_vlines, streams=[control_vline_stream, samples_pipe])
        scatter_map = hv.DynamicMap(self.update_scatter, streams=[control_vline_stream, samples_pipe])

        self.streams = {
            "samples": samples_pipe,
            "vlines_control": control_vline_stream,
            "vlines": control_vline_stream
        }

        self.maps = {
            "samples": samples_map,
            "vlines_control": control_vline_map,
            "vlines": vlines_map,
            "scatter": scatter_map
        }

    def update_control_vlines(self, x1: float, x2: float, stroke_count: int, data):
        if None in [x1, x2]:
            x1, x2 = self.vlines_positions
        xs, _, cov = data
        xmin, xmax = np.min(xs), np.max(xs)
        bounds = (xmin, xmin, xmax, xmax)
        hline = hv.VLine(x=x1).opts(color=orange1)
        vline = hv.HLine(y=x2).opts(color=orange2)
        self.vlines_positions = (x1, x2)
        image = hv.Image(np.rot90(cov), bounds=bounds)
        points_3 = hv.Points([(x1, x1), (x2, x2), (x2, x1)]).opts(line_color=blue)
        point = hv.Points([(x1, x2)]).opts(fill_color=orange1, line_color="white")
        return image * hline * vline * points_3 * point

    def update_vlines(self, x1: float, x2: float, stroke_count: int, data: Optional[np.ndarray] = None):
        if None in [x1, x2]:
            x1, x2 = self.vlines_positions
        xs, ys, _ = data
        ids1 = np.argmin((xs - x1)**2)
        ids2 = np.argmin((xs - x2)**2)
        ones = np.ones_like(xs[ids1])
        points1 = hv.Points(data=(ones * x1, ys[ids1])).opts(color=orange1)
        points2 = hv.Points(data=(ones * x2, ys[ids2])).opts(color=orange2)
        vline1 = hv.VLine(x=x1).opts(color=orange1)
        vline2 = hv.VLine(x=x2).opts(color=orange2)
        return vline1 * vline2 * points1 * points2

    def update_scatter(self, x1: float, x2: float, stroke_count: int, data: Optional[np.ndarray] = None):
        if None in [x1, x2]:
            x1, x2 = self.vlines_positions
        xs, ys, _ = data
        ids1 = np.argmin((xs - x1)**2)
        ids2 = np.argmin((xs - x2)**2)
        y1 = ys[ids1]
        y2 = ys[ids2]
        ys = np.stack([y1, y2], axis=1)
        return hv.Scatter(data=ys)

    def update_samples(self, data: Tuple[np.ndarray, np.ndarray]):
        xs, ys, _ = data
        return hv.Path((xs, ys))

    def view(self):
        data = self.data
        xmin, xmax = np.min(data), np.max(data)
        x1_dim = hv.Dimension('x₁', range=(xmin, xmax))
        x2_dim = hv.Dimension('x₂', range=(xmin, xmax))

        samples_map = self.maps["samples"]
        samples_map = samples_map.opts(width=600, height=350, show_grid=True, padding=(0, 0.1), toolbar=None)
        samples_map = samples_map.opts(opts.Path(color=blue, framewise=True))
        samples_map = samples_map.redim.label(y="f(x)", x="x")

        control_vline_map = self.maps["vlines_control"]
        control_vline_map = control_vline_map.redim(x=x1_dim, y=x2_dim)
        control_vline_map = control_vline_map.opts(show_grid=True)
        control_vline_map = control_vline_map.opts(opts.HLine(line_width=2),
                                                   opts.VLine(line_width=2),
                                                   opts.Points(color="white", marker="s", size=8),
                                                   opts.Image(cmap="viridis"))

        vlines_map = self.maps["vlines"]
        vlines_map = vlines_map.opts(toolbar=None)
        vlines_map = vlines_map.opts(opts.VLine(line_width=2), opts.Points(size=6))

        scatter_map = self.maps["scatter"]
        scatter_map = scatter_map.redim.label(y="f(x₂)", x="f(x₁)")
        scatter_map = scatter_map.opts(padding=(0.5, 0.5),
                                       show_grid=True,
                                       toolbar=None)
        scatter_map = scatter_map.opts(opts.Scatter(size=7, framewise=True, fill_color=orange1, line_color=orange2))

        title = pn.pane.Markdown("## GP samples visualization", max_height=25)
        descr = pn.pane.Markdown(
            "_For moving x₁ and x₂ bars:_ <br>_1. turn off **pan** tool,_ <br>_2. click and move the blue dot_")
        row0 = pn.Row(pn.Spacer(width=25), self.kernels_controller.view())
        row1 = samples_map * vlines_map
        row2 = pn.Row(control_vline_map, scatter_map)
        return pn.Column(title, descr, row0, row1, row2)
