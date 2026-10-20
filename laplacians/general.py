import numpy as np
from copy import deepcopy
import dionysus as dio
from tqdm.auto import tqdm

def gudhi_to_dio(st, decimals=10):
    f = dio.Filtration()
    for vertices, time in st.get_filtration():
        f.append(dio.Simplex(vertices, round(time, decimals)))
    return f

def time_to_idx(t, relevant_times):
    """
    Returns the index of the time t in the relevant_times array.
    If t is not in relevant_times, returns the index of the largest time less than t.
    """
    idx = np.searchsorted(relevant_times, t, side='right') - 1
    if idx < 0:
        raise ValueError(f"Time {t} is before the start of the filtration.")
    return idx

def compute_boundary_matrices(f: dio.Filtration, weight_fun = lambda x: 1, compute_n_simplices_to_time = False):
    """
    Computes the boundary matrices for a given filtration f.

    Args:
        f (dio.Filtration): The filtration for which to compute the boundary matrices.
        weight_fun (function, optional): A function that takes a simplex name and returns its weight. Defaults to a function that returns 1 for all simplices.

    Returns:
        boundary_matrices (list of np.ndarray): The boundary matrices for each dimension.
        simplices_at_time (function): A function that takes a time t and returns the number of simplices at that time for each dimension.
        relevant_times (np.ndarray): The times at which the filtration changes.
    """
    f.sort()

    t_old = 0
    relevant_times = []
    simplices_at_time = []
    n_simplicies_seen_total = []
    if compute_n_simplices_to_time:
        n_simplices_to_time = {}

    for s in f:
        if s.data != t_old:
            relevant_times.append(t_old)
            simplices_at_time.append(deepcopy(n_simplicies_seen_total))
            t_old = s.data

        q = s.dimension()
        while q >= len(n_simplicies_seen_total):
            n_simplicies_seen_total.append(0)
        n_simplicies_seen_total[q] += 1

        if compute_n_simplices_to_time:
                n_simplices_to_time.setdefault(q, {})[n_simplicies_seen_total[q]] = len(relevant_times)

    # To be safe add a q+1 entry, in case of cycles appearing in q+1, but not boundaries.
    n_simplicies_seen_total.append(0)

    # Clean up n_simplicies_seen_per_time to be same length at every time
    relevant_times.append(s.data)
    simplices_at_time.append(deepcopy(n_simplicies_seen_total))
    maxq = len(simplices_at_time[-1])
    for qi in range(len(simplices_at_time)):
        simplices_at_time[qi] = simplices_at_time[qi] + [0]*(maxq-len(simplices_at_time[qi]))
    
    relevant_times = np.array(relevant_times)
    simplices_at_end = simplices_at_time[-1]

    boundary_matrices = [np.zeros((simplices_at_end[max(0,q-1)], simplices_at_end[q])) for q in range(maxq)]
    name_to_idx = [{} for _ in range(maxq)]
    simplex_bar = tqdm(range(len(f)), leave=False, desc=f"Computing boundary matrices")
    for s in f:
        simplex_bar.update(1)
        q = s.dimension()

        name = str(s).split("<")[1].split(">")[0]
        idx = len(name_to_idx[q])
        name_to_idx[q][name] = idx

        if q > 0:
            for i, bdry_simplex in enumerate(s.boundary()):
                name_bdry_simplex = str(bdry_simplex).split("<")[1].split(">")[0]  
                idx_bdry_simplex = name_to_idx[q-1][name_bdry_simplex]

                # Boundaries as described in OG (Dawson) paper
                boundary_matrices[q][idx_bdry_simplex, idx] = weight_fun(name)/weight_fun(name_bdry_simplex)*(-1)**i

                # Assuming product weights
                # for v in s:
                #     if v not in bdry_simplex:
                #         name_bdry_simplex = str(v)
                #         break
                # boundary_matrices[q][idx_bdry_simplex, idx] = weight_fun(name_bdry_simplex)*(-1)**i

    simplex_bar.close()

    if compute_n_simplices_to_time:
        return boundary_matrices, simplices_at_time, relevant_times, n_simplices_to_time
    else:
        return boundary_matrices, simplices_at_time, relevant_times
