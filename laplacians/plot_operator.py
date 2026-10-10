import numpy as np
import dionysus as dio
import pandas as pd
import matplotlib.pyplot as plt

from .laplacians import combinatorial_Laplacian_eigenvalues
from .general import time_to_idx, compute_boundary_matrices

def plot_persistence_diagrams(dgms, show=False, labels=False, ax=None,
                 line_style=None, pt_style=None,
                 limits = None, plot_limits = None):
    """
    Plot persistence diagrams for multiple dimensions.

    Arguments:
        dgm (Diagram): See for example `init_diagrams`.

    Keyword Arguments:
        show (bool): Display the plot. (Default: False)
        labels (bool): Set axis labels. (Default: False)
        ax (AxesSubplot): Axes that should be used for plotting (Default: None)
        pt_style (dict): argments passed to `ax.scatter` for style of points.
        line_style (dict): argments passed to `ax.plot` for style of diagonal line.
    """

    import matplotlib.pyplot as plt

    inf = float('inf')
    min_birth,max_birth,min_death,max_death = inf,-1,inf,-1
    for dgm_i, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        if not limits:
            min_birth = min(min_birth, min((p.birth for p in dgm))-1)
            max_birth = max(max_birth, max((p.birth for p in dgm)))

            for p in dgm:
                if p.death != inf:
                    max_death = max(max_death, max((p.death for p in dgm if p.death != inf))+1)
                    break
            if max_death == -1:
                max_death = max_birth + 1
        else:
            min_birth, max_birth, min_death, max_death = limits

    done_dims = []

    for dgm_i, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        line_kwargs = {}
        pt_kwargs = {}
        if pt_style is not None:
            pt_kwargs.update(pt_style)
        if line_style is not None:
            line_kwargs.update(line_style)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        # ax.set_aspect('equal', 'datalim')

        min_diag = min_birth
        max_diag = max_death

        pt_kwargs["c"] = {0: "b", 1: "r", 2: "g", 3: "y"}[dgm_i]

        ax.scatter([p.birth for p in dgm], [p.death if p.death != inf else max_death for p in dgm], **pt_kwargs)
        

        if labels:
            ax.set_xlabel('birth')
            ax.set_ylabel('death')

        ## clip the view
        if plot_limits is not None:
            ax.set_xlim(plot_limits[0], plot_limits[1])
            ax.set_ylim(plot_limits[2], plot_limits[3])
        
        done_dims.append(dgm_i)
        #plt.axes().set_xlim([min_birth, max_birth])
        #plt.axes().set_ylim([min_death, max_death])

    ax.plot([min_diag, max_diag], [max_diag, max_diag], c="black")
    ax.set_xlim([min_diag, max_diag])
    ax.set_ylim([min_diag, max_diag])
    y_tickslabels = list(ax.get_yticklabels())
    y_tickslabels[-1] = "$\infty$"
    y_ticks = list(ax.get_yticks())

    ax.set_ylim([min_diag, max_diag*1.1])
    ax.set_yticklabels(y_tickslabels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tickslabels)

    ax.set_xlabel("birth")
    ax.set_ylabel("death")

    ax.legend([f"dim {dgm_i}" for dgm_i in done_dims])

    ax.plot([min_diag, max_diag], [min_diag, max_diag], **line_kwargs)

        # line for infinity

    if show:
        plt.show()

def eig_plot_helper(x, fun, use_pos = True, eps = 1e-10):
    """
    Helper function for plotting eigenvalues.
    
    Args:
        x (list or np.ndarray): The eigenvalues to process.
        fun (function): The function to apply to the eigenvalues.
        use_pos (bool, optional): Whether to use only positive eigenvalues. Defaults to True.
        eps (float, optional): The threshold for considering an eigenvalue as positive. Defaults to 1e-10.
    
    Returns:
        float: The result of applying the function to the eigenvalues.
    """
    if len(x) > 0:
        if not use_pos:
            return fun(x)
        else:
            pos_x = x[np.abs(x)>eps]
            if len(pos_x) > 0:
                return fun(pos_x)
            else:
                return 0
    else:
        return np.nan

def plot_eigenvalues(eigenvalues, relevant_times, plot_types = ["min"], plot_type_to_fun = {}, 
                     f = None, integer_time_steps = False,
                     plot_args_mesh = {}, plot_args_diag = {}, plot_args_line = {},
                     plot_value_on_diagonal = False, show=True, fill_up_figure = False, tolerance = 1e-10):
    """
    Plots the calculated eigenvalues in a colormesh.

    Args:
        eigenvalues (dict): A dictionary containing eigenvalues at each combination of start and end times for each dimension.
        relevant_times (np.ndarray): An array of relevant times in the filtration.
        plot_types (list or str, optional): A list of plot types to generate or "all" for all types. Entries should be keys in plot_type_to_fun. Defaults to "all".
        f (dio.Filtration, optional): The filtration used to compute the eigenvalues. Required for plotting persistence points in the colormesh. If None, no points are plotted. Defaults to None.
        integer_time_steps (bool, optional): Whether to use integer time steps on the axes. Defaults to False.
        plot_args_mesh (dict, optional): Additional arguments for the mesh plot. Defaults to {}.
        plot_args_diag (dict, optional): Additional arguments for the diagonal points plot. Defaults to {}.
        plot_args_line (dict, optional): Additional arguments for the diagonal line plot. Defaults to {}.
        plot_type_to_fun (dict, optional): A dictionary containing types of functions to plot. Each key should correspond to a dictionary with keys "function" (Function to plot), "title" (Title in plot) and "use_pos" (Whether to use only positive eigenvalues (default >1e-10)).
        plot_value_on_diagonal (bool, optional): Whether to plot the diagonal values as a line plot over the colormesh. Defaults to False.

    Returns:
        fig, ax: The matplotlib figure and axes objects.
    """
    plot_type_to_fun = {
        "min": {"function": np.min, "title": "Min eigenvalue", "use_pos": True},
        "max": {"function": np.max, "title": "Max eigenvalue", "use_pos": False},
        "sum": {"function": np.sum, "title": "Sum of eigenvalues", "use_pos": False},
        "mean": {"function": np.mean, "title": "Mean non-zero eigenvalues", "use_pos": True},
        "product": {"function": np.prod, "title": "Prod non-zero eigenvalues", "use_pos": True},
        "n_non_zero": {"function": len, "title": "Amount of non-zero eigenvalues", "use_pos": True},
        "n_zero": {"function": lambda x: np.count_nonzero(x<1e-10), "title": "Amount of zero eigenvalues", "use_pos": False},
    } | plot_type_to_fun

    plot_args_mesh = {"alpha": 1, "cmap": "jet"} | plot_args_mesh
    plot_args_diag = {"c": "white", "marker": "o", "alpha": 1, "edgecolor": "black"} | plot_args_diag
    plot_args_line = {"c": "r", "alpha": 0.5} | plot_args_line

    max_dim = max(eigenvalues.keys())

    fig, ax = plt.subplots(max(max_dim+1, 2), len(plot_types))
    # if max_dim == 1:
    #     ax = 
    fig.set_size_inches(4*len(plot_types), 4*(max_dim+1))
    if len(plot_types) == 1:
        cur_ax = lambda x,y: ax[x]
    else:
        cur_ax = lambda x,y: ax[x, y]
        

    fig.gca()

    if fill_up_figure and f is not None:
        boundary_matrices, simplices_at_time, relevant_times = compute_boundary_matrices(f)
        for q in eigenvalues.keys():
            for s in eigenvalues[q].keys():
                s_i = time_to_idx(s, relevant_times)
                if simplices_at_time[s_i][q] > 0:
                    for t in eigenvalues[q][s].keys():
                        if t > s and len(eigenvalues[q][s][t]) == 0:
                            eigenvalues[q][s][t] = np.array([0])

    # For pcolormesh, we need a grid that is one bigger as every rectangle needs starting and ending points.
    # Now added a point the same distance away as the last two real points.
    relevant_times = relevant_times.tolist()
    extended_relevant_times = relevant_times+[2*relevant_times[-1]-relevant_times[-2]]
    x, y = np.meshgrid(extended_relevant_times, extended_relevant_times)

    # Compute persistence points if filtration is given.
    if f is not None:
        p = dio.cohomology_persistence(f, 47, True)
        dgms = dio.init_diagrams(p, f)
        barcodes_births, barcodes_deaths = [], []
        for q in range(len(dgms)):
            barcodes_births.append([])
            barcodes_deaths.append([])
            for p in dgms[q]:
                barcodes_births[q].append(p.birth)
                if p.death != np.inf:
                    barcodes_deaths[q].append(p.death)
                else:
                    barcodes_deaths[q].append(extended_relevant_times[-1])

    for q in range(max_dim+1):
        df_evals = pd.DataFrame(eigenvalues[q]).apply(lambda x: x.apply(lambda y: np.array([]) if type(y) == float else y))
        for ax_i, plot_type in enumerate(plot_types):
            # For product of eigenvalues, use log scale
            if plot_type == "product":
                plot_args_mesh["norm"] = "log"
            else:
                plot_args_mesh["norm"] = "linear"

            # Plot colormesh.
            plot_fun, plot_title, use_pos = plot_type_to_fun[plot_type]["function"], plot_type_to_fun[plot_type]["title"], plot_type_to_fun[plot_type]["use_pos"]
            im = cur_ax(q, ax_i).pcolormesh(x, y, df_evals.apply(lambda x: x.apply(lambda y: eig_plot_helper(y, plot_fun, use_pos=use_pos, eps=tolerance))).values, **plot_args_mesh)
            plt.colorbar(im, ax=cur_ax(q, ax_i), pad=0.15)
            if q == 0:
                cur_ax(q, ax_i).set_title(f"{plot_title}")
            if q == max_dim:
                cur_ax(q, ax_i).set_xlabel("s")

            # Plot persistence points.
            if f is not None:
                cur_ax(q, ax_i).scatter(barcodes_births[q], barcodes_deaths[q], **plot_args_diag)

            # Plot diagonal.
            if plot_value_on_diagonal:
                ax_n = cur_ax(q, ax_i).twinx()
                ax_n.plot([relevant_times[0]] + [val for val in relevant_times[1:] for _ in (0, 1)], [eig_plot_helper(eigenvalues[q][t][t], plot_fun, use_pos=use_pos, eps=tolerance) for t in relevant_times for _ in (0,1)][:-1], **plot_args_line)
                if plot_type == "product":
                    ax_n.set_yscale("log")

            if integer_time_steps:
                cur_ax(q, ax_i).yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        cur_ax(q, 0).set_ylabel(f"q={q}\n\nt")
    fig.tight_layout()

    if show:
        fig.show()
    return fig, ax

def plot_Laplacian_diagram(eigenvalues, plot_type = "min", plot_type_to_fun = {}, 
                           f=None, dgms=None, show=False, min_dim=0, max_dim=1,
                           line_style={}, pt_style_pers={}, pt_style_lap={}, return_points=False):
    """
    Plots a trace diagram with persistence points and eigenvalue points.

    Args:
        eigenvalues (dict): A dictionary containing eigenvalues at each combination of start and end times for each dimension.
        f (dio.Filtration, optional): The filtration used to compute the persistence points. If None, dgms must be provided for points. Otherwise no persistence points in diagram. Defaults to None.
        dgms (list of dio.Diagram, optional): Precomputed persistence diagrams. If None, they will be computed from f. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to False.
        min_dim (int, optional): The minimum dimension to plot. Defaults to 0.
        max_dim (int, optional): The maximum dimension to plot. Defaults to 1.
        line_style (dict, optional): Additional style arguments for the diagonal line. Defaults to {}.
        pt_style_normal (dict, optional): Additional style arguments for the persistence points. Defaults to {}.
        lap_pt_style (dict, optional): Additional style arguments for the eigenvalue points. Defaults to {}.
        title (string): Title of plot.

    Returns:
        s_lists (list of lists): s values of each of the points in the diagram for each dimension.
        t_lists (list of lists): t values of each of the points in the diagram for each dimension.
        cmaps (list of lists): Sum of eigenvalues at each of the points in the diagram for each dimension.
    """
    plot_type_to_fun = {
        "min": {"function": np.min, "title": "Min eigenvalue", "use_pos": True},
        "max": {"function": np.max, "title": "Max eigenvalue", "use_pos": False},
        "sum": {"function": np.sum, "title": "Sum of eigenvalues", "use_pos": False},
        "mean": {"function": np.mean, "title": "Mean non-zero eigenvalues", "use_pos": True},
        "product": {"function": np.prod, "title": "Prod non-zero eigenvalues", "use_pos": True},
        "n_non_zero": {"function": len, "title": "Amount of non-zero eigenvalues", "use_pos": True},
        "n_zero": {"function": lambda x: np.count_nonzero(x<1e-10), "title": "Amount of zero eigenvalues", "use_pos": False},
    } | plot_type_to_fun
    
    assert plot_type in plot_type_to_fun
    plot_fun, plot_title, use_pos = plot_type_to_fun[plot_type]["function"], plot_type_to_fun[plot_type]["title"], plot_type_to_fun[plot_type]["use_pos"]

    fig, ax = plt.subplots(1, max_dim+1-min_dim, figsize=(6*(max_dim+1-min_dim),5))
    if f is not None:
        p = dio.cohomology_persistence(f, 47, True)
        dgms = dio.init_diagrams(p, f)

    s_lists = [[]]*(max_dim + 1)
    t_lists = [[]]*(max_dim + 1)
    cmaps = [[]]*(max_dim + 1)

    line_style = {} | line_style
    pt_style_pers =  {"c": "black", "marker": "x", "alpha": 0.5, "s": 100} | pt_style_pers
    pt_style_lap = {"cmap": "jet"} | pt_style_lap

    for q in range(min_dim, max_dim+1):
        if max_dim+1-min_dim > 1:
            cur_ax = ax[q-min_dim]
        else:
            cur_ax = ax

        s_list = []
        t_list = []
        cmap = []
        for s in eigenvalues[q].keys():
            for t in eigenvalues[q][s].keys():
                fun_output = eig_plot_helper(eigenvalues[q][s][t], plot_fun, use_pos=use_pos)
                if np.abs(fun_output) > 1e-10:
                    s_list.append(s)
                    t_list.append(t)
                    cmap.append(fun_output)
        s_lists[q] = s_list
        t_lists[q] = t_list
        cmaps[q] = cmap

        if len(s_list) > 0:
            min_diag = 0
            max_diag = np.max(t_list)*1.1
        else:
            min_diag = -1
            max_diag = 2

        cur_ax.set_aspect('equal', 'datalim')

        if dgms is not None:
            dgm = dgms[q]
            cur_ax.scatter([p.birth for p in dgm], [p.death if p.death != np.inf else max_diag for p in dgm], **pt_style_pers)

        scatter_im = cur_ax.scatter(s_list, t_list, c=cmap, **pt_style_lap)
        plt.colorbar(scatter_im, ax=cur_ax, pad=0.15)

        # Line y=x
        cur_ax.plot([min_diag, max_diag], [min_diag, max_diag], **line_style)
        # Line for infinity
        cur_ax.plot([min_diag, max_diag], [max_diag, max_diag], c="black")

        cur_ax.set_xlabel('birth')
        cur_ax.set_ylabel('death')
        cur_ax.set_title(f"{plot_title} (dim {q})")

    if show:
        fig.show()
    else:
        plt.close(fig)
    if return_points:
        return s_lists, t_lists, cmaps
    return fig, ax

def plot_combinatorial_eigenvalues(eigenvalues = None, relevant_times = None, f: dio.Filtration = None,
                                   plot_types = ["min"], plot_type_to_fun = {}, max_dim=1, show=True):
    """
    Plots the eigenvalues of the combinatorial Laplacian at all times.

    Args:
        eigenvalues (dict, optional): A dictionary containing eigenvalues at each time for each dimension q. If None, it will be computed from the filtration f. Defaults to None.
        relevant_times (list, optional): A list of relevant times in the filtration. If None, it will be computed from the filtration f. Defaults to None.
        f (dio.Filtration, optional): The filtration used to compute the eigenvalues. If eigenvalues and relevant_times are not provided, they will be computed from this filtration. Defaults to None.
        eval_fun (function, optional): A function to compute a summary statistic of the eigenvalues (e.g., np.min, np.mean). Defaults to np.min.
        max_dim (int, optional): The maximum dimension q to plot. Defaults to 1.
        labels (list of functions, optional): A list of functions to generate labels for the plots. If None, default labels will be used. Defaults to
    """

    assert (eigenvalues is not None and relevant_times is not None) or f is not None, "eigenvalues and relevant_times or filtration must be provided"
    if f is not None:
        eigenvalues, relevant_times = combinatorial_Laplacian_eigenvalues(f=f, max_dim=max_dim)

    # assert plot_type in plot_type_to_fun
    # plot_fun, plot_title, use_pos = plot_type_to_fun[plot_type]["function"], plot_type_to_fun[plot_type]["title"], plot_type_to_fun[plot_type]["use_pos"]

    fig, ax = plt.subplots(max_dim+1, 1, figsize=(10, 5*(max_dim+1)), sharex=True)
    fig.tight_layout()
    ax = ax.flatten() if max_dim > 0 else [ax]

    labels = [lambda q : fr'Agg($Spec(\Delta_{q}^t)$)', lambda q : fr'$\beta_{q}$']

    for q in range(max_dim+1):
        x_axis = []
        y_lap = [[] for _ in plot_types]
        y_betti = []

        for t_i, t in enumerate(relevant_times):
            lap_labels = []
            for type_i, plot_type in enumerate(plot_types):
                plot_fun, plot_title, use_pos = plot_type_to_fun[plot_type]["function"], plot_type_to_fun[plot_type]["title"], plot_type_to_fun[plot_type]["use_pos"]
                y_lap[type_i].append(eig_plot_helper(eigenvalues[q][t][t], plot_fun, use_pos=use_pos))
                lap_labels.append(plot_title)

            x_axis.append(t)
            cur_betti = eig_plot_helper(eigenvalues[q][t][t], lambda x: np.count_nonzero(x<1e-10), use_pos=False)
            y_betti.append(cur_betti)

            # For step plot, add point again, but with next x value.
            if t_i < len(relevant_times) - 1:
                x_axis.append(relevant_times[t_i+1])
                for type_i, plot_type in enumerate(plot_types):
                    plot_fun, plot_title, use_pos = plot_type_to_fun[plot_type]["function"], plot_type_to_fun[plot_type]["title"], plot_type_to_fun[plot_type]["use_pos"]
                    y_lap[type_i].append(eig_plot_helper(eigenvalues[q][t][t], plot_fun, use_pos=use_pos))
                y_betti.append(cur_betti)
                
            # zero_evals.append(cur_evals[mesh_zero])
            # pos_evals.append(cur_evals[~mesh_zero])
        
        ax_betti = ax[q].twinx()
        p2, = ax_betti.plot(x_axis, y_betti, label=labels[1](q), color='red', linestyle='--')
        plots = [p2]
        for type_i, plot_type in enumerate(plot_types):
            p1, = ax[q].plot(x_axis, y_lap[type_i], label=lap_labels[type_i])
            plots.append(p1)
        
        # ax[q].plot(relevant_times, [eval_fun(evals) if len(evals) > 0 else None for evals in pos_evals], label=f'q={q} (laplacian)', color='blue', linestyle='--')
        # ax_betti.plot(relevant_times, [len(evals) for evals in zero_evals], label=f'q={q} (betti)', color='orange')
        
        ax[q].set_title(f"q={q}")
        ax[q].set_ylim(0, np.max([np.nanmax(y_lap[type_i])*1.1 for type_i in range(len(plot_types))]))
        ax[q].set_ylabel(labels[0](q), fontsize=20)
        # ax[q].yaxis.label.set_color(p1.get_color())

        ax_betti.set_ylabel(labels[1](q), fontsize=20)
        ax_betti.set_ylim(0, np.nanmax(y_betti)+1)
        ax_betti.set_yticks(range(0, int(np.nanmax(y_betti))+2))
        ax_betti.yaxis.label.set_color(p2.get_color())

        ax[q].legend(handles=plots, loc='upper left')
        # ax_betti.legend()

    ax[-1].set_xlabel('Time', fontsize=20)

    if show:
        fig.show()

    return fig, ax