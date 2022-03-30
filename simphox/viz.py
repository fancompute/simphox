import numpy as np
from typing import Dict

import xarray
from .typing import Tuple, Optional, List

try:
    HOLOVIEWS_IMPORTED = True
    import holoviews as hv
    from holoviews.streams import Pipe
    from holoviews import opts
    import panel as pn
    from bokeh.models import Range1d, LinearAxis
    from bokeh.models.renderers import GlyphRenderer
    from bokeh.plotting.figure import Figure
except ImportError:
    HOLOVIEWS_IMPORTED = False

try:
    K3D_IMPORTED = True
    import k3d
    from k3d import Plot
except ImportError:
    K3D_IMPORTED = False

from matplotlib import colors as mcolors


def _plot_twinx_bokeh(plot, _):
    """Hook to plot data on a secondary (twin) axis on a Holoviews Plot with Bokeh backend.

    Args:
        plot: Holoviews plot object to hook for twinx

    See Also:
        The code was copied from a comment in https://github.com/holoviz/holoviews/issues/396.
        - http://holoviews.org/user_guide/Customizing_Plots.html#plot-hooks
        - https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html#twin-axes

    """
    fig: Figure = plot.state
    glyph_first: GlyphRenderer = fig.renderers[0]  # will be the original plot
    glyph_last: GlyphRenderer = fig.renderers[-1]  # will be the new plot
    right_axis_name = "twiny"
    # Create both axes if right axis does not exist
    if right_axis_name not in fig.extra_y_ranges.keys():
        # Recreate primary axis (left)
        y_first_name = glyph_first.glyph.y
        y_first_min = glyph_first.data_source.data[y_first_name].min()
        y_first_max = glyph_first.data_source.data[y_first_name].max()
        y_first_offset = (y_first_max - y_first_min) * 0.1
        fig.y_range = Range1d(
            start=y_first_min - y_first_offset,
            end=y_first_max + y_first_offset
        )
        fig.y_range.name = glyph_first.y_range_name
        # Create secondary axis (right)
        y_last_name = glyph_last.glyph.y
        y_last_min = glyph_last.data_source.data[y_last_name].min()
        y_last_max = glyph_last.data_source.data[y_last_name].max()
        y_last_offset = (y_last_max - y_last_min) * 0.1
        fig.extra_y_ranges = {right_axis_name: Range1d(
            start=y_last_min - y_last_offset,
            end=y_last_max + y_last_offset
        )}
        fig.add_layout(LinearAxis(y_range_name=right_axis_name, axis_label=glyph_last.glyph.y), "right")
    # Set right axis for the last glyph added to the figure
    glyph_last.y_range_name = right_axis_name


def get_extent_2d(shape, spacing: Optional[float] = None):
    """2D extent

    Args:
        shape: shape of the elements to plot
        spacing: spacing between grid points (assumed to be isotropic)

    Returns:
        The extent in 2D.

    """
    return (0, shape[0] * spacing, 0, shape[1] * spacing) if spacing else (0, shape[0], 0, shape[1])


def plot_eps_2d(ax, eps: np.ndarray, spacing: Optional[float] = None, cmap: str = 'gray'):
    """Plot eps in 2D

    Args:
        ax: Matplotlib axis handle
        eps: epsilon permittivity
        spacing: spacing between grid points (assumed to be isotropic)
        cmap: colormap for field array (we highly recommend RdBu)

    """
    extent = get_extent_2d(eps.shape, spacing)
    if spacing:  # in microns!
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.imshow(eps.T, cmap=cmap, origin='lower', alpha=1, extent=extent)


def plot_field_2d(ax, field: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                  cmap: str = 'RdBu', mat_cmap: str = 'gray', alpha: float = 0.8, vmax = None):
    """Plot field in 2D

    Args:
        ax: Matplotlib axis handle
        field: field to plot
        eps: epsilon permittivity for overlaying field onto materials
        spacing: spacing between grid points (assumed to be isotropic)
        cmap: colormap for field array (we highly recommend RdBu)
        mat_cmap: colormap for eps array (we recommend gray)
        alpha: transparency of the plots to visualize overlay

    """
    extent = get_extent_2d(field.shape, spacing)
    if spacing:  # in microns!
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_xlabel(r'$x$ ($\mu$m)')
    if eps is not None:
        plot_eps_2d(ax, eps, spacing, mat_cmap)
    im_val = field
    vmax = np.max(im_val * np.sign(field.flat[np.abs(field).argmax()])) if vmax is None else vmax
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    ax.imshow(im_val.T, cmap=cmap, origin='lower', alpha=alpha, extent=extent, norm=norm)


def plot_eps_1d(ax, eps: Optional[np.ndarray], spacing: Optional[float] = None,
                color: str = 'blue', units: str = "$\mu$m", axis_label_rotation: float = 90):
    """Plot eps in 1D.

    Args:
        ax: Matplotlib axis handle
        eps: epsilon permittivity for overlaying field onto materials
        spacing: spacing between grid points (assumed to be isotropic)
        color: Color to plot the epsilon
        units: Units for plotting (default microns)
        axis_label_rotation: Rotate the axis label in case a plot is made with shared axes.

    """
    x = np.arange(eps.shape[0]) * spacing
    if spacing:
        ax.set_xlabel(rf'$x$ ({units})')
        ax.set_ylabel(rf'Relative permittivity ($\epsilon$)', color=color,
                      rotation=axis_label_rotation, labelpad=15)
    ax.plot(x, eps, color=color)
    ax.tick_params(axis='y', labelcolor=color)


def plot_field_1d(ax, field: np.ndarray, field_name: str, eps: Optional[np.ndarray] = None,
                  spacing: Optional[float] = None, color: str = 'red', eps_color: str = 'blue',
                  units: str = "$\mu$m"):
    """Plot field in 1D

    Args:
        ax: Matplotlib axis handle.
        field: Field to plot.
        field_name: Name of the field being plotted
        spacing: spacing between grid points (assumed to be isotropic).
        color: Color to plot the epsilon
        units: Units for plotting (default microns)

    """
    x = np.arange(field.shape[0]) * spacing
    if spacing:  # in microns!
        ax.set_xlabel(rf'$x$ ({units})')
        ax.set_ylabel(rf'{field_name}', color=color)
    ax.plot(x, field.real, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    if eps is not None:
        ax_eps = ax.twinx()
        plot_eps_1d(ax_eps, eps, spacing, eps_color, units, axis_label_rotation=270)


def hv_field_1d(field: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                width: float = 600):
    x = np.arange(field.shape[0]) * spacing
    field = field.squeeze().real / np.max(np.abs(field))
    c1 = hv.Curve((x, (field + 1) / 2), kdims='x', vdims='field').opts(
        width=width, show_grid=True, framewise=True, yaxis='left', ylim=(-1, 1))
    c2 = hv.Curve((x, eps), kdims='x', vdims='eps').opts(width=width, show_grid=True, framewise=True, color='red',
                                                         hooks=[_plot_twinx_bokeh])
    return c1 * c2


def hv_field_2d(field: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                cmap: str = 'RdBu', mat_cmap: str = 'gray', alpha: float = 0.2, width: float = 600):
    extent = get_extent_2d(field.squeeze().T.shape, spacing)
    bounds = (extent[0], extent[2], extent[1], extent[3])
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    field_img = hv.Image(field.squeeze().T.real / np.max(np.abs(field)),
                         bounds=bounds, vdims='field').opts(cmap=cmap, aspect=aspect, frame_width=width)
    eps_img = hv.Image(eps.T / np.max(eps), bounds=bounds).opts(cmap=mat_cmap, alpha=alpha, aspect=aspect, frame_width=width)
    return field_img.redim.range(field=(-1, 1)) * eps_img


def hv_power_1d(power: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                width: float = 600):
    x = np.arange(power.shape[0]) * spacing
    power = power.squeeze().real / np.max(np.abs(power))
    c1 = hv.Curve((x, power), kdims='x', vdims='field').opts(width=width, show_grid=True, framewise=True,
                                                             yaxis='left', ylim=(-1, 1))
    c2 = hv.Curve((x, eps), kdims='x', vdims='eps').opts(width=width, show_grid=True, framewise=True, color='red',
                                                         hooks=[_plot_twinx_bokeh])
    return c1 * c2


def hv_power_2d(power: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                cmap: str = 'hot', mat_cmap: str = 'gray', alpha: float = 0.2, width: float = 600):
    extent = get_extent_2d(power.squeeze().T.shape, spacing)
    bounds = (extent[0], extent[2], extent[1], extent[3])
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    power_img = hv.Image(power.squeeze().T.real / np.max(np.abs(power)),
                         bounds=bounds, vdims='power').opts(cmap=cmap, aspect=aspect, frame_width=width)
    eps_img = hv.Image(eps.T / np.max(eps), bounds=bounds).opts(cmap=mat_cmap, alpha=alpha,
                                                                aspect=aspect, frame_width=width)
    return power_img.redim.range(power=(0, 1)) * eps_img


def plot_power_2d(ax, power: np.ndarray, eps: Optional[np.ndarray] = None, spacing: Optional[float] = None,
                  cmap: str = 'hot', mat_cmap: str = 'gray', alpha: float = 0.8, vmax = None):
    """Plot the power (computed using Poynting) in 2D

    Args:
        ax: Matplotlib axis handle
        power: power array of size (X, Y)
        eps: epsilon for overlay with materials
        spacing: spacing between grid points (assumed to be isotropic)
        cmap: colormap for power array
        mat_cmap: colormap for eps array (we recommend gray)
        alpha: transparency of the plots to visualize overlay

    """
    extent = get_extent_2d(power.shape, spacing)
    if spacing:  # in microns!
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_xlabel(r'$x$ ($\mu$m)')
    if eps is not None:
        plot_eps_2d(ax, eps, spacing, mat_cmap)
    vmax = np.max(power) if vmax is None else vmax
    ax.imshow(power.T, cmap=cmap, origin='lower', alpha=alpha, extent=extent, vmax=vmax)


def plot_power_3d(plot: "Plot", power: np.ndarray, eps: Optional[np.ndarray] = None, axis: int = 0,
                  spacing: float = 1, color_range: Tuple[float, float] = None, alpha: float = 100,
                  samples: float = 1200):
    """Plot the 3d power in a notebook given the fields :math:`E` and :math:`H`.

    Args:
        plot: K3D plot handle (NOTE: this is for plotting in a Jupyter notebook)
        power: power (either Poynting field of size (3, X, Y, Z) or power of size (X, Y, Z))
        eps: permittivity (if specified, plot with default options)
        axis: pick the correct axis if inputting power in Poynting field
        spacing: spacing between grid points (assumed to be isotropic)
        color_range: color range for visualization (if none, use half maximum value of field)
        alpha: alpha for k3d plot
        samples: samples for k3d plot rendering

    Returns:

    """

    if not K3D_IMPORTED:
        raise ImportError("Need to install k3d for this function to work.")

    power = power[axis] if power.ndim == 4 else power
    color_range = (0, np.max(power) / 2) if color_range is None else color_range

    if eps is not None:
        plot_eps_3d(plot, eps, spacing=spacing)  # use defaults for now

    power_volume = k3d.volume(
        power.transpose((2, 1, 0)),
        alpha_coef=alpha,
        samples=samples,
        color_range=color_range,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.hot).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        name='power'
    )

    bounds = [0, power.shape[0] * spacing, 0, power.shape[1] * spacing, 0, power.shape[2] * spacing]
    power_volume.transform.bounds = bounds
    plot += power_volume


def plot_field_3d(plot: "Plot", field: np.ndarray, eps: Optional[np.ndarray] = None, axis: int = 1,
                  imag: bool = False, spacing: float = 1,
                  alpha: float = 100, samples: float = 1200, color_range: Tuple[float, float] = None):
    """

    Args:
        plot: K3D plot handle (NOTE: this is for plotting in a Jupyter notebook)
        field: field to plot
        eps: permittivity (if specified, plot with default options)
        axis: pick the correct axis for power in Poynting vector form
        imag: whether to use the imaginary (instead of real) component of the field
        spacing: spacing between grid points (assumed to be isotropic)
        color_range: color range for visualization (if none, use half maximum value of field)
        alpha: alpha for k3d plot
        samples: samples for k3d plot rendering

    Returns:

    """

    if not K3D_IMPORTED:
        raise ImportError("Need to install k3d for this function to work.")

    field = field[axis] if field.ndim == 4 else field
    field = field.imag if imag else field.real
    color_range = np.asarray((0, np.max(field)) if color_range is None else color_range)

    if eps is not None:
        plot_eps_3d(plot, eps, spacing=spacing)  # use defaults for now

    bounds = [0, field.shape[0] * spacing, 0, field.shape[1] * spacing, 0, field.shape[2] * spacing]

    pos_e_volume = k3d.volume(
        volume=field.transpose((2, 1, 0)),
        alpha_coef=alpha,
        samples=samples,
        color_range=color_range,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.RdBu).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        name='pos'
    )

    neg_e_volume = k3d.volume(
        volume=-field.transpose((2, 1, 0)),
        alpha_coef=alpha,
        samples=1200,
        color_range=color_range,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.RdBu_r).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        name='neg'
    )

    neg_e_volume.transform.bounds = bounds
    pos_e_volume.transform.bounds = bounds

    plot += neg_e_volume
    plot += pos_e_volume


def plot_eps_3d(plot: "Plot", eps: Optional[np.ndarray] = None, spacing: float = 1,
                color_range: Tuple[float, float] = None, alpha: float = 100, samples: float = 1200):
    """

    Args:
        plot: K3D plot handle (NOTE: this is for plotting in a Jupyter notebook)
        eps: relative permittivity
        spacing: spacing between grid points (assumed to be isotropic)
        color_range: color range for visualization (if none, use half maximum value of field)
        alpha: alpha for k3d plot
        samples: samples for k3d plot rendering

    Returns:

    """

    if not K3D_IMPORTED:
        raise ImportError("Need to install k3d for this function to work.")

    color_range = (1, np.max(eps)) if color_range is None else color_range

    eps_volume = k3d.volume(
        eps.transpose((2, 1, 0)),
        alpha_coef=alpha,
        samples=samples,
        color_map=(np.array(k3d.colormaps.matplotlib_color_maps.Greens).reshape(-1, 4)).astype(np.float32),
        compression_level=8,
        color_range=color_range,
        name='epsilon'
    )

    bounds = [0, eps.shape[0] * spacing, 0, eps.shape[1] * spacing, 0, eps.shape[2] * spacing]
    eps_volume.transform.bounds = bounds
    plot += eps_volume


def scalar_metrics_viz(metric_config: Dict[str, List[str]]):
    if not HOLOVIEWS_IMPORTED:
        raise ImportError("Holoviews not imported, cannot visualize")
    metrics_pipe = {title: Pipe(data=xarray.DataArray(
        data=np.asarray([[] for _ in metric_config[title]]),
        coords={
            'metric': metric_config[title],
            'iteration': np.arange(0)
        },
        dims=['metric', 'iteration'],
        name=title
    )) for title in metric_config}
    metrics_dmaps = [
        hv.DynamicMap(lambda data: hv.Dataset(data).to(hv.Curve, kdims=['iteration']).overlay('metric'),
                      streams=[metrics_pipe[title]]).opts(opts.Curve(framewise=True, shared_axes=False, title=title))
        for title in metric_config
    ]
    return pn.Row(*metrics_dmaps), metrics_pipe
