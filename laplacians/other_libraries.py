import petls
import numpy as np
from copy import deepcopy
import dionysus as dio
from.general import compute_boundary_matrices, time_to_idx, gudhi_to_dio
from .multiplicity import standard_algorithm, lowest_nonzero_row, return_pos_evals
import gudhi as gd
from scipy.sparse import coo_matrix

# def gudhi_to_petls(st):
#     # petls.Alpha only supports 3D points (CGAL Alpha_complex_3d), so build a petls.Complex from a gudhi simplex tree instead.
#     # We don't use petls.Complex(simplex_tree=st): its helper orders columns by iterating a python set, which breaks
#     # the correspondence between boundary columns and filtration values for larger complexes.
#     top_dim = st.dimension()
#     indices = [{} for _ in range(top_dim+1)]
#     filtrations = [[] for _ in range(top_dim+1)]
#     for simplex, t in st.get_filtration():
#         q = len(simplex) - 1
#         indices[q][tuple(simplex)] = len(filtrations[q])
#         filtrations[q].append(t)
#     boundaries = []
#     for q in range(1, top_dim+1):
#         row, col, data = [], [], []
#         for simplex, j in indices[q].items():
#             for k in range(q+1):
#                 row.append(indices[q-1][simplex[:k] + simplex[k+1:]])
#                 col.append(j)
#                 data.append((-1)**k)
#         boundaries.append(coo_matrix((data, (row, col)), shape=(len(filtrations[q-1]), len(filtrations[q]))).toarray())
#     return petls.Complex(boundaries, filtrations)

def petls_output_to_eigenvalues(spectra, eigenvalues = None, round_to = 5):
    if eigenvalues is None:
        eigenvalues = {}
    relevant_times = []
    for spectrum in spectra:
        if round_to is not None:
            s = np.round(spectrum[1], round_to)
            t = np.round(spectrum[2], round_to)
        else:
            s = spectrum[1]
            t = spectrum[2]
        relevant_times.append(s)
        relevant_times.append(t)
        eigenvalues.setdefault(spectrum[0], {})
        eigenvalues[spectrum[0]].setdefault(s, {})
        eigenvalues[spectrum[0]][s][t] = np.array(spectrum[3])
    relevant_times = np.array(sorted(list(set(relevant_times))))
    return eigenvalues, relevant_times

def petls_combined(points, threshold, max_dim, f = None, petls_complex=None, round_to = 5, fill_eigenvalues = False, filtration_type = "rips"):
    if f is None:
        if filtration_type == "rips":
            f = dio.fill_rips(points, max_dim, threshold)
        elif filtration_type == "alpha":
            st = gd.AlphaComplex(points=points).create_simplex_tree()
            f = gudhi_to_dio(st)
        else:
            raise ValueError("Invalid filtration type. Choose from 'rips' or 'alpha'.")
    boundary_matrices, simplices_at_time, relevant_times, n_simplices_to_time = compute_boundary_matrices(f, compute_n_simplices_to_time=True)

    if fill_eigenvalues:
        downLaps = {}

    request_list = []
    for q in range(max_dim+1):
        if len(boundary_matrices) <= q+1 or boundary_matrices[q+1].shape[1] == 0:
            continue
        # n_simplices_to_time[q+1][0] = relevant_times[0]

        R, V = standard_algorithm(boundary_matrices[q+1])
        new_chain_complexes = {}

        for i in range(V.shape[1]):
            s_i = n_simplices_to_time[q][max(lowest_nonzero_row(R[:, i]), 1)]
            t_i = n_simplices_to_time[q+1][i + 1]
            new_chain_complexes[(s_i, t_i)] = 1

        to_calculate = set()
        for (s_i, t_i) in new_chain_complexes.keys():
            to_calculate.add((s_i, t_i))
            for (s_prime, t_prime) in new_chain_complexes.keys():
                if (s_prime > s_i and t_prime < t_i) or (s_prime < s_i and t_prime > t_i):
                    to_calculate.add((max(s_i, s_prime), max(t_i, t_prime)))
        to_calculate = sorted(list(to_calculate), key=lambda x: (x[0], x[1]))
        for (s_i, t_i) in to_calculate:
            request_list.append((q, relevant_times[s_i], relevant_times[t_i]))

        if fill_eigenvalues:
                    used_for_downLap_helper = {}
                    for (s_i, t_i) in to_calculate:
                        if t_i not in used_for_downLap_helper:
                            used_for_downLap_helper[t_i] = s_i
                        else:
                            used_for_downLap_helper[t_i] = max(used_for_downLap_helper[t_i], s_i)
                    used_for_downLap = {}
                    for i in range(len(relevant_times)):
                        if i in used_for_downLap_helper:
                            used_for_downLap.setdefault((used_for_downLap_helper[i], i), []).append(i)
                        else:
                            j = i
                            while j>=0:
                                if j in used_for_downLap_helper:
                                    used_for_downLap.setdefault((used_for_downLap_helper[j], j), []).append(i)
                                    break
                                j -= 1
                    downLaps[q] = {tuple(v): k for k, v in used_for_downLap.items()}

    if round_to is not None:
        eigenvalues = {q: {np.round(s, round_to): {np.round(t, round_to): np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    else:
        eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}

    if petls_complex is None:
        if filtration_type == "rips":
            petls_complex = petls.Rips(points = points, threshold=threshold)
        elif filtration_type == "alpha":
            try:
                petls_complex = petls.Alpha(points = points, max_dim=max_dim)
            except:
                petls_complex = petls.Complex(simplex_tree=st)
        else:
            raise ValueError("Invalid filtration type. Choose from 'rips' or 'alpha'.")
    eigpairs = petls_complex.eigenpairs(request_list=request_list)
    eigenvalues, relevant_times = petls_output_to_eigenvalues(eigpairs, eigenvalues=eigenvalues, round_to=round_to)

    if fill_eigenvalues:
        if round_to is not None:
            tol = np.pow(10, -round_to)
        else:
            tol = 1e-5
        new_eigenvalues = deepcopy(eigenvalues)
        for q in range(max_dim+1):
            for s in sorted(list(eigenvalues[q].keys())):
                for t in sorted(list(eigenvalues[q][s].keys())):
                    if len(eigenvalues[q][s][t]) > 0:
                        done = False
                        for next_t in sorted(list(eigenvalues[q][s].keys())):
                            if done:
                                break
                            if next_t < t:
                                continue
                            for next_s in sorted(list(eigenvalues[q].keys())):
                                if next_s < s:
                                    continue
                                if next_s == s and next_t == t:
                                    continue
                                if next_s > next_t:
                                    break
                                if len(eigenvalues[q][next_s][next_t]) > 0:
                                    if next_s == s:
                                        done = True
                                    break
                                else:
                                    new_eigenvalues[q][next_s][next_t] = eigenvalues[q][s][t]
        eigenvalues = new_eigenvalues
        for q in range(max_dim, -1, -1):
            if q-1 in downLaps:
                for s in eigenvalues[q].keys():
                    s_i = time_to_idx(s, relevant_times)
                    for k in downLaps[q-1].keys():
                        if s_i in k:
                            down_evals = None   #return_pos_evals(eigenvalues[q-1][relevant_times[downLaps[q-1][k][0]]][relevant_times[downLaps[q-1][k][1]]], tol=tol)
                            for t in eigenvalues[q][s].keys():
                                if len(eigenvalues[q][s][t]) > 0:
                                    continue
                                elif t >= s:
                                    if down_evals is None:
                                        down_evals = return_pos_evals(np.array(petls_complex.eigenpairs(dim=q, a=s, b=s)[0]), tol=tol)
                                    try:
                                        zero_evals = np.zeros(simplices_at_time[s_i][q] - len(down_evals))
                                    except:
                                        print(f"Error: simplices_at_time = {simplices_at_time[s_i]}, s_i = {s_i}, q = {q}, down_evals = {down_evals}")
                                        raise
                                    eigenvalues[q][s][t] = np.concatenate((down_evals, zero_evals))


    return eigenvalues, relevant_times
