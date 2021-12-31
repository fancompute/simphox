from collections import defaultdict

from .utils import fix_dataclass_init_docs
from .sim import SimGrid
from .typing import Optional, Callable, Union, List, Tuple, Dict

import jax
import jax.numpy as jnp
from jax.config import config
from jax.experimental.optimizers import adam
import numpy as np
import dataclasses
import xarray

try:
    HOLOVIEWS_IMPORTED = True
    import holoviews as hv
    from holoviews.streams import Pipe
    import panel as pn
except ImportError:
    HOLOVIEWS_IMPORTED = False

from .viz import scalar_metrics_viz

config.parse_flags_with_absl()


@fix_dataclass_init_docs
@dataclasses.dataclass
class OptProblem:
    """An optimization problem

    An optimization problem consists of a neural network defined at least by input parameters :code:`rho`,
    the transform function :code:`T(rho)` (:math:`T(\\rho(x, y))`) (default identity),
    and objective function :code:`C(T(rho))` (:math:`C(T(\\rho(x, y)))`), which maps to a scalar.
    For use with an inverse design problem (the primary use case in this module), the user can include an
    FDFD simulation and a source (to be fed into the FDFD solver). The FDFD simulation and source are then
    used to define a function :code:`S(eps) == S(T(rho))` that solves the FDFD problem
    where `eps == T(rho)` (:math:`\\epsilon(x, y)) := T(\\rho(x, y))`),
    in which case the objective function evaluates :code:`C(S(T(rho)))`
    (:math:`\\epsilon(x, y)) := C(S(T((\\rho(x, y))))`).

    Args:
        transform_fn: The JAX-transformable transform function to yield epsilon (identity if None,
                    must be a single :code:`transform_fn` (to be broadcast to all)
                    or a list to match the FDFD objects respectively). Examples of transform_fn
                    could be smoothing functions, symmetry functions, and more (which can be compounded appropriately).
        cost_fn: The JAX-transformable cost function (or tuple of such functions)
            corresponding to src that takes in output of solve_fn from :code:`opt_solver`.
        sim: SimGrid(s) used to generate the solver (FDFD is not run is :code:`fdfd` is :code:`None`)
        source: A numpy array source (FDFD is not run is :code:`source` is :code:`None`)
        metrics_fn: A metric_fn that returns useful dictionary data based on fields and FDFD object
         at certain time intervals (specified in opt). Each problem is supplied this metric_fn
         (Optional, ignored if :code:`None`).

    """
    transform_fn: Callable
    cost_fn: Callable
    sim: SimGrid
    source: str
    metrics_fn: Optional[Callable[[np.ndarray, SimGrid], Dict]] = None

    def __post_init__(self):
        self.fn = self.sim.get_sim_sparams_fn(self.source, self.transform_fn)\
            if self.source is not None else self.transform_fn


@fix_dataclass_init_docs
@dataclasses.dataclass
class OptViz:
    """An optimization visualization object

    An optimization visualization object consists of a plot for monitoring the
    history and current state of an optimization in real time.

    Args:
        cost_dmap: Cost dynamic map for streaming cost fn over time
        simulations_panel: Simulations panel for visualizing simulation results from last iteration
        costs_pipe: Costs pipe for streaming cost fn over time
        simulations_pipes: Simulations pipes of form :code:`eps, field, power`
            for visualizing simulation results from last iteration
        metrics_panels: Metrics panels for streaming metrics over time for each simulation (e.g. powers/power ratios)
        metrics_pipes: Metrics pipes for streaming metrics over time for each simulation
        metric_config: Metric config (a dictionary that describes how to plot/group the real-time metrics)

    """
    cost_dmap: "hv.DynamicMap"
    simulations_panels: Dict[str, "pn.layout.Panel"]
    costs_pipe: "Pipe"
    simulations_pipes: Dict[str, Tuple["Pipe", "Pipe", "Pipe"]]
    metric_config: Optional[Dict[str, List[str]]] = None
    metrics_panels: Optional[Dict[str, "hv.DynamicMap"]] = None
    metrics_pipes: Optional[Dict[str, Dict[str, "Pipe"]]] = None


@fix_dataclass_init_docs
@dataclasses.dataclass
class OptRecord:
    """An optimization record

    We need an object to hold the history, which includes a list of costs (we avoid the term loss
    as it may be related to  denoted

    Attributes:
        costs: List of costs
        params: Params (:math:`\rho`) transformed into the design
        metrics: An xarray for metrics with dimensions :code:`name`, :code:`metric`, :code:`iteration`
        eps: An xarray for relative permittivity with dimensions :code:`name`, :code:`x`, :code:`y`
        fields: An xarray for a selected field component with dimensions :code:`name`, :code:`x`, :code:`y`

    """
    costs: np.ndarray
    params: jnp.ndarray
    metrics: xarray.DataArray
    eps: xarray.DataArray
    fields: xarray.DataArray


def opt_run(opt_problem: Union[OptProblem, List[OptProblem]], init_params: np.ndarray, num_iters: int,
            pbar: Optional[Callable] = None, step_size: float = 1, viz_interval: int = 0, metric_interval: int = 0,
            viz: Optional[OptViz] = None, backend: str = 'cpu',
            eps_interval: int = 0, field_interval: int = 0) -> OptRecord:
    """Run the optimization.

    The optimization can be done over multiple simulations as long as those simulations
    share the same set of params provided by :code:`init_params`.

    Args:
        opt_problem: An :code:`OptProblem` or list of :code:`OptProblem`'s. If a list is provided,
            the optimization optimizes the sum of all objective functions.
            If the user wants to weight the objective functions, weights must be inlcuded in the objective function
            definition itself, but we may provide support for this feature at a later time if needed.
        init_params: Initial parameters for the optimizer (:code:`eps` if :code:`None`)
        num_iters: Number of iterations to run
        pbar: Progress bar to keep track of optimization progress with ideally a simple tqdm interface
        step_size: For the Adam update, specify the step size needed.
        viz_interval: The optimization intermediate results are recorded every :code:`record_interval` steps
            (default of 0 means do not visualize anything)
        metric_interval: The interval over which a recorded object (e.g. metric, param)
         are recorded in a given :code:`OptProblem` (default of 0 means do not record anything).
        viz: The :code:`OptViz` object required for visualizing the optimization in real time.
        backend: Recommended backend for :code:`ndim == 2` is :code:`'cpu'` and :code:`ndim == 3` is :code:`'gpu'`
        eps_interval: Whether to record the eps at the specified :code:`eps_interval`.
            Beware, this can use up a lot of memory during the opt so use judiciously.
        field_interval: Whether to record the field at the specified :code:`field_interval`.
            Beware, this can use up a lot of memory during the opt so use judiciously.

    Returns:
        A tuple of the final eps distribution (:code:`transform_fn(p)`) and parameters :code:`p`

    """

    opt_init, opt_update, get_params = adam(step_size=step_size)
    opt_state = opt_init(init_params)

    # define opt_problems
    opt_problems = [opt_problem] if isinstance(opt_problem, OptProblem) else opt_problem
    n_problems = len(opt_problems)

    # opt problems that include both an FDFD sim and a source sim
    sim_opt_problems = [op for op in opt_problems if op.sim is not None and op.source is not None]

    if viz is not None:
        if not len(viz.simulations_pipes) == len(sim_opt_problems):
            raise ValueError("Number of viz_pipes must match number of opt problems")

    # Define the simulation and objective function acting on parameters rho
    solve_fn = [None if (op.source is None or op.sim is None) else op.fn for op in opt_problems]

    def overall_cost_fn(rho: jnp.ndarray):
        evals = [op.cost_fn(s(rho)) if s is not None else op.cost_fn(rho) for op, s in zip(opt_problems, solve_fn)]
        return jnp.array([obj for obj, _ in evals]).sum() / n_problems, [aux for _, aux in evals]

    # Define a compiled update step
    def step_(current_step, state):
        vaux, g = jax.value_and_grad(overall_cost_fn, has_aux=True)(get_params(state))
        v, aux = vaux
        return v, opt_update(current_step, g, state), aux

    def _update_eps(state):
        rho = get_params(state)
        for op in opt_problems:
            op.sim.eps = np.asarray(jax.lax.stop_gradient(op.transform_fn(rho)))

    step = jax.jit(step_, backend=backend)

    iterator = pbar(range(num_iters)) if pbar is not None else range(num_iters)

    costs = []
    history = defaultdict(list)

    for i in iterator:
        v, opt_state, data = step(i, opt_state)
        _update_eps(opt_state)
        for sop, sparams_fields in zip(sim_opt_problems, data):
            sim = sop.sim
            sparams, e, h = sim.decorate(*sparams_fields)
            hz = np.asarray(h[2]).squeeze().T
            if viz_interval > 0 and i % viz_interval == 0 and viz is not None:
                eps_pipe, field_pipe, power_pipe = viz.simulations_pipes[sim.name]
                eps_pipe.send((sim.eps.T - np.min(sim.eps)) / (np.max(sim.eps) - np.min(sim.eps)))
                field_pipe.send(hz.real / np.max(hz.real))
                power = np.abs(hz) ** 2
                power_pipe.send(power / np.max(power))
            if metric_interval > 0 and i % metric_interval == 0 and viz is not None:
                metrics = sop.metrics_fn(sparams)
                for metric_name, metric_value in metrics.items():
                    history[f'{metric_name}/{sop.sim.name}'].append(metric_value)
                for title in viz.metrics_pipes[sop.sim.name]:
                    viz.metrics_pipes[sop.sim.name][title].send(
                        xarray.DataArray(
                            data=np.asarray([history[f'{metric_name}/{sop.sim.name}']
                                             for metric_name in viz.metric_config[title]]),
                            coords={
                                'metric': viz.metric_config[title],
                                'iteration': np.arange(i + 1)
                            },
                            dims=['metric', 'iteration'],
                            name=title
                        )
                    )
            if eps_interval > 0 and i % eps_interval == 0:
                history[f'eps/{sop.sim.name}'].append((i, sop.sim.eps))
            if field_interval > 0 and i % field_interval == 0:
                history[f'field/{sop.sim.name}'].append((i, hz.T))
        iterator.set_description(f"𝓛: {v:.5f}")
        costs.append(jax.lax.stop_gradient(v))
        if viz is not None:
            viz.costs_pipe.send(np.asarray(costs))
    _update_eps(opt_state)

    all_metric_names = sum([metric_names for _, metric_names in viz.metric_config.items()], [])
    metrics = xarray.DataArray(
        data=np.array([[history[f'{metric_name}/{sop.sim.name}']
                          for metric_name in all_metric_names] for sop in sim_opt_problems]),
        coords={
            'name': [sop.sim.name for sop in sim_opt_problems],
            'metric': all_metric_names,
            'iteration': np.arange(num_iters)
        },
        dims=['name', 'metric', 'iteration'],
        name='metrics'
    ) if sim_opt_problems and metric_interval != 0 else []
    eps = xarray.DataArray(
        data=np.array([[eps for _, eps in history[f'eps/{sop.sim.name}']] if eps_interval > 0 else []
                         for sop in sim_opt_problems]),
        coords={
            'name': [sop.sim.name for sop in sim_opt_problems],
            'iteration': [it for it, _ in history[f'eps/{sim_opt_problems[0].sim.name}']],
            'x': np.arange(sim_opt_problems[0].sim.shape[0]),
            'y': np.arange(sim_opt_problems[0].sim.shape[1]),
        },
        dims=['name', 'iteration', 'x', 'y'],
        name='eps'
    ) if sim_opt_problems and eps_interval != 0 else []
    fields = xarray.DataArray(
        data=np.asarray([[field for _, field in history[f'field/{sop.sim.name}']] if field_interval > 0 else []
                         for sop in sim_opt_problems]),
        coords={
            'name': [sop.sim.name for sop in sim_opt_problems],
            'iteration': [it for it, _ in history[f'field/{sim_opt_problems[0].sim.name}']],
            'x': np.arange(sim_opt_problems[0].sim.shape[0]),
            'y': np.arange(sim_opt_problems[0].sim.shape[1]),
        },
        dims=['name', 'iteration', 'x', 'y'],
        name='fields'
    ) if sim_opt_problems and field_interval != 0 else []
    return OptRecord(costs=np.asarray(costs), params=get_params(opt_state), metrics=metrics, eps=eps, fields=fields)


def opt_viz(opt_problem: Union[OptProblem, List[OptProblem]], metric_config: Dict[str, List[str]]) -> OptViz:
    """Optimization visualization panel

    Args:
        opt_problem: An :code:`OptProblem` or list of :code:`OptProblem`'s.
        metric_config: A dictionary of titles mapped to lists of metrics to plot in the graph (for overlay)

    Returns:
        A tuple of visualization panel, loss curve pipe, and visualization pipes

    """
    opt_problems = [opt_problem] if isinstance(opt_problem, OptProblem) else opt_problem
    viz_panel_pipes = {op.sim.name: op.sim.viz_panel()
                       for op in opt_problems if op.sim is not None and op.source is not None}
    costs_pipe = Pipe(data=[])

    metrics_panel_pipes = {op.sim.name: scalar_metrics_viz(metric_config=metric_config)
                           for op in opt_problems if op.sim is not None and op.source is not None}

    return OptViz(
        cost_dmap=hv.DynamicMap(hv.Curve, streams=[costs_pipe]).opts(title='Cost Fn (𝓛)'),
        simulations_panels={name: v[0] for name, v in viz_panel_pipes.items()},
        costs_pipe=costs_pipe,
        simulations_pipes={name: v[1] for name, v in viz_panel_pipes.items()},
        metrics_panels={name: m[0] for name, m in metrics_panel_pipes.items()},
        metrics_pipes={name: m[1] for name, m in metrics_panel_pipes.items()},
        metric_config=metric_config
    )
