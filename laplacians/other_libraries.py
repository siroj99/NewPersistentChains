import petls
import numpy as np

def petls_output_to_eigenvalues(spectra):
    eigenvalues = {}
    relevant_times = []
    for spectrum in spectra:
        spectrum[1] = np.round(spectrum[1], 5)
        spectrum[2] = np.round(spectrum[2], 5)
        relevant_times.append(spectrum[1])
        relevant_times.append(spectrum[2])
        eigenvalues.setdefault(spectrum[0], {})
        eigenvalues[spectrum[0]].setdefault(spectrum[1], {})
        eigenvalues[spectrum[0]][spectrum[1]][spectrum[2]] = np.array(spectrum[3])
    relevant_times = np.array(sorted(list(set(relevant_times))))
    return eigenvalues, relevant_times

import dionysus as dio
from.general import compute_boundary_matrices
from .multiplicity import standard_algorithm, lowest_nonzero_row

def petls_Rips_combined(points, threshold, max_dim):
    f = dio.fill_rips(points, max_dim, threshold)
    boundary_matrices, simplices_at_time, relevant_times, n_simplices_to_time = compute_boundary_matrices(f, compute_n_simplices_to_time=True)

    request_list = []
    for q in range(max_dim+1):
        if boundary_matrices[q+1].shape[1] == 0:
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
            request_list.append((q, s_i, t_i))

    spectra = petls.Rips(points = points, threshold=threshold).spectra(request_list=request_list)
    eigenvalues, relevant_times = petls_output_to_eigenvalues(spectra)
    return eigenvalues, relevant_times
