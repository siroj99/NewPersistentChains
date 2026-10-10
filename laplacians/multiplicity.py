import array
from math import comb
from matplotlib.pylab import LinAlgError
import numpy as np
from copy import deepcopy
import dionysus as dio
import scipy
import scipy.stats as ss
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gudhi as gd

from .general import compute_boundary_matrices, time_to_idx

import time

def calc_pinv(q, s, t, boundary_matrices, simplices_at_time, inverse_method="compute_inverse"):
    """
    Calculates the pseudo inverse of B22 directly.
    """
    if simplices_at_time(t)[q+1] == simplices_at_time(s)[q+1]:
        return np.zeros((0,0))
    elif simplices_at_time(t)[q] == simplices_at_time(s)[q]:
        return 0
    else:
        if inverse_method == "compute_inverse":
            B22 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
            return np.linalg.pinv(B22)@B22
        elif inverse_method == "least_squares":
            B22 = boundary_matrices[q+1][simplices_at_time(s)[q]:simplices_at_time(t)[q], simplices_at_time(s)[q+1]:simplices_at_time(t)[q+1]]
            return np.linalg.lstsq(B22, B22, rcond=None)[0]



# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv NEW vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

def lowest_nonzero_row(v, tol=1e-10):
    """
    Returns the index of the lowest non-zero row in a vector v.
    If all entries are zero, returns 0.
    """
    non_zero_indices = np.where(np.abs(v) > tol)[0]
    return non_zero_indices[-1]+1 if non_zero_indices.size > 0 else 0

def gram_schmidt_added(new_vectors, prev_basis=None):
    """
    Applies the Gram-Schmidt process to a set of vectors, with a known basis B.
    
    Args:
        B (optional, array-like): The known basis vectors.
        new_vectors (array-like): The new vectors to orthogonalize.
                          
    Returns:
        np.ndarray: A 2D numpy array containing the resulting basis vectors.
    """
    # Convert to a float array to avoid integer division issues
    # new_vectors = np.array(new_vectors, dtype=float)
    orthogonal = np.zeros_like(new_vectors)
    
    for i in range(new_vectors.shape[1]):
        v = new_vectors[:, i]

        # subtract the projection of v onto all previously computed basis vectors in B
        if prev_basis is not None:
            v = v - prev_basis @ (prev_basis.T @ v)
        
        # Subtract the projection of v onto all previously computed basis vectors
        for j in range(i):
            u_j = orthogonal[:, j]

            # Projection formula: (v · u_j / u_j · u_j) * u_j
            projection = (np.dot(v, u_j)) * u_j
            v = v - projection
            
        orthogonal[:, i] = v / np.linalg.norm(v) 
        
    # if normalize:
    #     # Calculate the magnitude (L2 norm) of each row vector
    #     norms = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    #     # Remove zero norm vectors from orthogonal
    #     orthogonal = orthogonal[norms[:, 0] > 1e-10]
    #     norms = norms[norms[:, 0] > 1e-10]
    #     return orthogonal / norms
        
    return orthogonal

def standard_algorithm(B, tol=1e-10, max_iter=1000):
    R = deepcopy(B)
    V = np.eye(B.shape[1])
    low_i_set = {}
    for j in tqdm(range(B.shape[1]), leave=False, desc="Standard algorithm"):
        for _ in range(max_iter):
            low_R_j = lowest_nonzero_row(R[:, j], tol=tol) - 1
            if low_R_j <= -1:
                break
            if low_R_j not in low_i_set:
                low_i_set[low_R_j] = j
                break
            if low_R_j in low_i_set:
                k = low_i_set[low_R_j]
                V[:, j] = V[:, j] - (R[low_R_j, j] / R[low_R_j, k]) * V[:, k]
                R[:, j] = R[:, j] - (R[low_R_j, j] / R[low_R_j, k]) * R[:, k]
    return R, V

def return_pos_evals(eigenvalues, tol=1e-10):
    return np.array([ev for ev in eigenvalues if ev > tol])

def complete_multiplicity_operator_eigenvalues(boundary_matrices = None, simplices_at_time = None, relevant_times = None, f: dio.Filtration = None, 
                                        weight_fun = lambda x: 1, min_dim = 0, max_dim = 1, verb = False, up_only = False, fill_eigenvalues = False):
    """
    Computes the eigenvalues of the persistent multiplicity operator for each combination of start and end times in a filtration, given Multiplicity function.
    NOTE: Either give boundary_matrices, simplices_at_time and relevant_times or give filtration f.

    Args:
        boundary_matrices (list of np.ndarray, optional): List of boundary matrices for each dimension.
        simplices_at_time (function, optional): Function that returns the number of simplices at a given time for each dimension.
        relevant_times (list, optional): List of relevant times in the filtration.
        f (dio.Filtration, optional): Filtration from which to compute boundary matrices if not provided.
        weight_fun (function, optional): Function to compute weights for simplices.
        min_dim (int, optional): Minimum dimension to compute eigenvalues for.
        max_dim (int, optional): Maximum dimension to compute eigenvalues for.
        Multiplicity_fun (function, optional): Function to compute the multiplicity operator given projection matrices
        compute_only_trace (bool, optional): If True, only compute the trace of the multiplicity operator instead of all eigenvalues. Reduces computation time slightly.

    Returns:
        eigenvalues (dict): Dictionary containing eigenvalues for each dimension and time pair.
        relevant_times (list): List of relevant times used in the computation.
    """


    # Either boundary matrices, simplices at time and relevant times or filtration f must be provided.
    assert (boundary_matrices is not None and simplices_at_time is not None and relevant_times is not None) or (f is not None)

    if f is not None:
        f.sort()
        boundary_matrices, simplices_at_time, relevant_times, n_simplices_to_time = compute_boundary_matrices(f, weight_fun, compute_n_simplices_to_time=True)
    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(min_dim, max_dim+1)}

    basis_time = 0
    matrix_eig_time = 0

    if not up_only:
        downLaps = {}

    for q in range(min_dim, max_dim+1):
        if boundary_matrices[q+1].shape[1] == 0:
            continue

        # n_simplices_to_time[q+1][0] = relevant_times[0]

        R, V = standard_algorithm(boundary_matrices[q+1])
        new_chain_complexes = {}

        for i in range(V.shape[1]):
            s_i = n_simplices_to_time[q][max(lowest_nonzero_row(R[:, i]), 1)]
            t_i = n_simplices_to_time[q+1][i + 1]
            if (s_i, t_i) not in new_chain_complexes:
                new_chain_complexes[(s_i, t_i)] = V[:, i][None].T/np.linalg.norm(V[:, i])
            else:
                new_chain_complexes[(s_i, t_i)] = np.column_stack((new_chain_complexes[(s_i, t_i)], V[:, i][None].T/np.linalg.norm(V[:, i])))

        to_calculate = set()
        for (s_i, t_i) in new_chain_complexes.keys():
            to_calculate.add((s_i, t_i))
            for (s_prime, t_prime) in new_chain_complexes.keys():
                if (s_prime > s_i and t_prime < t_i) or (s_prime < s_i and t_prime > t_i):
                    if verb:
                        print(f"Adding to to_calculate: ({max(s_i, s_prime)}, {max(t_i, t_prime)}) because of ({s_i}, {t_i}) and ({s_prime}, {t_prime})")
                    to_calculate.add((max(s_i, s_prime), max(t_i, t_prime)))

        # Sort the to_calculate set by s and then by t to ensure we process in the correct order
        to_calculate = sorted(list(to_calculate), key=lambda x: (x[0], x[1]))
        if verb:
            print(f"q: {q}, to_calculate: {to_calculate}")
        calculated_prev = {}

        if not up_only:
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
            if verb:
                print(f"q: {q}, downLaps: {downLaps[q]}")
                
        for (s_i, t_i) in tqdm(to_calculate):
            best_prev_lap = None

            # First look if straight down or straight left has been calculated:
            basis_t = None
            basis_s = None
            highest_prev_t = -1
            highest_prev_s = -1
            for (prev_s, prev_t) in calculated_prev.keys():
                if prev_t < t_i and prev_s == s_i:
                    if prev_t > highest_prev_t:
                        highest_prev_t = prev_t
                if prev_s < s_i and prev_t == t_i:
                    if prev_s > highest_prev_s:
                        highest_prev_s = prev_s
            if highest_prev_t != -1:
                basis_t, prev_lap_t = calculated_prev[(s_i, highest_prev_t)]

            if highest_prev_s != -1:
                basis_s, prev_lap_s = calculated_prev[(highest_prev_s, t_i)]

            B = boundary_matrices[q+1][:(simplices_at_time[s_i][q]), :]

            # We go over all situations.
            if basis_t is not None and basis_s is not None:
                if basis_t.shape[1] > basis_s.shape[1]:
                    best_basis = basis_t
                    best_prev_lap = prev_lap_t

                    other_basis = []
                    for (prev_s, prev_t) in new_chain_complexes.keys():
                        if prev_s < s_i and prev_t == t_i:
                            other_basis.append(new_chain_complexes[(prev_s, prev_t)])
                    # We know there have to be at least one such (prev_s, prev_t) because basis_s is not None.
                    other_basis = np.column_stack(other_basis) 
                else:
                    best_basis = basis_s
                    best_prev_lap = prev_lap_s

                    other_basis = []
                    for (prev_s, prev_t) in new_chain_complexes.keys():
                        if prev_s == s_i and prev_t < t_i:
                            other_basis.append(new_chain_complexes[(prev_s, prev_t)])
                    # We know there have to be at least one such (prev_s, prev_t) because basis_s is not None.
                    other_basis = np.column_stack(other_basis) 

                if (s_i, t_i) in new_chain_complexes:
                    other_basis = np.column_stack((other_basis, new_chain_complexes[(s_i, t_i)]))

                S = gram_schmidt_added(other_basis, prev_basis=best_basis)
                U = B@S
                Lap = U@U.T
                best_prev_lap = np.pad(best_prev_lap, ((0, Lap.shape[0] - best_prev_lap.shape[0]), (0, Lap.shape[1] - best_prev_lap.shape[1])), 'constant')
                Lap += best_prev_lap

                eigenvalues[q][relevant_times[s_i]][relevant_times[t_i]] = np.linalg.eigvalsh(Lap)

                calculated_prev[(s_i, t_i)] = (np.column_stack((best_basis, S)), Lap)
                if verb:
                    print(f"Calculating eigenvalues for (s_i, t_i) = ({s_i}, {t_i}) with previous basis in both directions.")
                    # print(f"new_chain_complexes:\n{new_chain_complexes[(s, t)]}\n")
                    
                    print(f"Orthonormal basis S:\n{S}\n")
                    
                    print(f"U:\n{U}\n")
                    
                    print(f"Lap:\n{Lap}\n")

                    
                    print(f"Best previous Laplacian:\n{best_prev_lap}\n")
                

            elif basis_t is not None:
                best_basis = basis_t
                best_prev_lap = prev_lap_t

                # (s,t) has to be in chain_complexes otherwise it would not have been in to_calculate
                
                S = gram_schmidt_added(new_chain_complexes[(s_i, t_i)], prev_basis=best_basis)
                
                U = B@S
                
                Lap = U@U.T
                

                best_prev_lap = np.pad(best_prev_lap, ((0, Lap.shape[0] - best_prev_lap.shape[0]), (0, Lap.shape[1] - best_prev_lap.shape[1])), 'constant')
                

                Lap += best_prev_lap

                eigenvalues[q][relevant_times[s_i]][relevant_times[t_i]] = np.linalg.eigvalsh(Lap)

                calculated_prev[(s_i, t_i)] = (np.column_stack((best_basis, S)), Lap)

                if verb:
                    print(f"Calculating eigenvalues for (s_i, t_i) = ({s_i}, {t_i}) with previous basis in t of size {best_basis.shape[1]}.")
                    print(f"new_chain_complexes:\n{new_chain_complexes[(s_i, t_i)]}\n")
                    print(f"Best previous basis:\n{best_basis}\n")
                    print(f"Orthonormal basis S:\n{S}\n")
                    print(f"U:\n{U}\n")
                    print(f"Lap:\n{Lap}\n")
                    print(f"Best previous Laplacian:\n{best_prev_lap}\n")

            elif basis_s is not None:
                best_basis = basis_s
                best_prev_lap = prev_lap_s

                # (s,t) has to be in chain_complexes otherwise it would not have been in to_calculate
                
                S = gram_schmidt_added(new_chain_complexes[(s_i, t_i)], prev_basis=best_basis)
                
                U = B@S
                
                Lap = U@U.T
                

                best_prev_lap = np.pad(best_prev_lap, ((0, Lap.shape[0] - best_prev_lap.shape[0]), (0, Lap.shape[1] - best_prev_lap.shape[1])), 'constant')
                

                Lap += best_prev_lap

                eigenvalues[q][relevant_times[s_i]][relevant_times[t_i]] = np.linalg.eigvalsh(Lap)

                calculated_prev[(s_i, t_i)] = (np.column_stack((best_basis, S)), Lap)

                if verb:
                    print(f"Calculating eigenvalues for (s_i, t_i) = ({s_i}, {t_i}) with previous basis in s of size {best_basis.shape[1]}.")
                    print(f"new_chain_complexes:\n{new_chain_complexes[(s_i, t_i)]}\n")
                    print(f"Best previous basis:\n{best_basis}\n")
                    print(f"Orthonormal basis S:\n{S}\n")
                    print(f"U:\n{U}\n")
                    print(f"Lap:\n{Lap}\n")
                    print(f"Best previous Laplacian:\n{best_prev_lap}\n")

            elif basis_t is None and basis_s is None:
                best_basis = None

                (best_prev_s_s, best_prev_s_t), (best_prev_t_s, best_prev_t_t) = (-1, -1), (-1, -1)
                for (prev_s, prev_t) in calculated_prev.keys():
                    if prev_s <= s_i and prev_t <= t_i:
                        if prev_s > best_prev_s_s:
                            (best_prev_s_s, best_prev_s_t) = (prev_s, prev_t)
                        if prev_s == best_prev_s_s and prev_t > best_prev_s_t:
                            (best_prev_s_s, best_prev_s_t) = (prev_s, prev_t)
                        if prev_t > best_prev_t_t:
                            (best_prev_t_s, best_prev_t_t) = (prev_s, prev_t)
                        if prev_t == best_prev_t_t and prev_s > best_prev_t_s:
                            (best_prev_t_s, best_prev_t_t) = (prev_s, prev_t)

                if best_prev_s_s != -1:
                    if best_prev_t_t != -1:
                        if calculated_prev[(best_prev_s_s, best_prev_s_t)][0].shape[1] > calculated_prev[(best_prev_t_s, best_prev_t_t)][0].shape[1]:
                            best_basis, best_prev_lap = calculated_prev[(best_prev_s_s, best_prev_s_t)]
                        else:
                            best_basis, best_prev_lap = calculated_prev[(best_prev_t_s, best_prev_t_t)]
                    else:
                        best_basis, best_prev_lap = calculated_prev[(best_prev_s_s, best_prev_s_t)]
                elif best_prev_t_t != -1:
                    best_basis, best_prev_lap = calculated_prev[(best_prev_t_s, best_prev_t_t)]

            
                if best_basis is None:
                    
                    # print(f"new_chain_complexes:\n{new_chain_complexes[(s, t)]}\n")
                    S = gram_schmidt_added(new_chain_complexes[(s_i, t_i)])
                    
                    U = B@S
                    
                    Lap = U@U.T
                    
                    eigenvalues[q][relevant_times[s_i]][relevant_times[t_i]] = np.linalg.eigvalsh(Lap)

                    calculated_prev[(s_i, t_i)] = (S, Lap)

                    if verb:
                        print(f"Calculating eigenvalues for (s_i, t_i) = ({s_i}, {t_i}) with no previous basis.")
                        print(f"Orthonormal basis S:\n{S}\n")
                        print(f"U:\n{U}\n")
                        print(f"Lap:\n{Lap}\n")
                else:
                    
                    S = gram_schmidt_added(new_chain_complexes[(s_i, t_i)], prev_basis=best_basis)
                    
                    U = B@S
                    
                    Lap = U@U.T
                    

                    best_prev_lap = np.pad(best_prev_lap, ((0, Lap.shape[0] - best_prev_lap.shape[0]), (0, Lap.shape[1] - best_prev_lap.shape[1])), 'constant')
                    

                    Lap += best_prev_lap

                    eigenvalues[q][relevant_times[s_i]][relevant_times[t_i]] = np.linalg.eigvalsh(Lap)

                    calculated_prev[(s_i, t_i)] = (np.column_stack((best_basis, S)), Lap)

                    if verb:
                        print(f"Calculating eigenvalues for (s_i, t_i) = ({s_i}, {t_i}) with previous basis diagonally of size {best_basis.shape[1]}.")
                        print(f"new_chain_complexes:\n{new_chain_complexes[(s_i, t_i)]}\n")
                        print(f"Best previous basis:\n{best_basis}\n")
                        print(f"Orthonormal basis S:\n{S}\n")
                        print(f"U:\n{U}\n")
                        print(f"Lap:\n{Lap}\n")
                        print(f"Best previous Laplacian:\n{best_prev_lap}\n")


    if not up_only and not fill_eigenvalues:
      for q in range(max_dim, min_dim-1, -1):
        if q-1 in downLaps:
            for s in eigenvalues[q].keys():
                s_i = time_to_idx(s, relevant_times)
                for k in downLaps[q-1].keys():
                    if s_i in k:
                        down_evals = return_pos_evals(eigenvalues[q-1][relevant_times[downLaps[q-1][k][0]]][relevant_times[downLaps[q-1][k][1]]])
                        for t in eigenvalues[q][s].keys():
                            if len(eigenvalues[q][s][t]) > 0:
                                eigenvalues[q][s][t].sort()
                                eigenvalues[q][s][t][:len(down_evals)] = down_evals
                            # elif s == t and simplices_at_time[s_i][q] > simplices_at_time[s_i-1][q]:
                            #     eigenvalues[q][s][t] = down_evals


    if fill_eigenvalues:
        new_eigenvalues = deepcopy(eigenvalues)
        for q in range(min_dim, max_dim+1):
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
        if not up_only:
            for q in range(max_dim, min_dim-1, -1):
                if q-1 in downLaps:
                    for s in eigenvalues[q].keys():
                        s_i = time_to_idx(s, relevant_times)
                        for k in downLaps[q-1].keys():
                            if s_i in k:
                                down_evals = return_pos_evals(eigenvalues[q-1][relevant_times[downLaps[q-1][k][0]]][relevant_times[downLaps[q-1][k][1]]])
                                for t in eigenvalues[q][s].keys():
                                    if len(eigenvalues[q][s][t]) > 0:
                                        up_evals = return_pos_evals(eigenvalues[q][s][t])
                                        zero_evals = np.zeros(simplices_at_time[s_i][q] - len(up_evals) - len(down_evals))
                                        eigenvalues[q][s][t] = np.concatenate((up_evals, down_evals, zero_evals))
                                    elif t >= s:
                                        zero_evals = np.zeros(simplices_at_time[s_i][q] - len(down_evals))
                                        eigenvalues[q][s][t] = np.concatenate((down_evals, zero_evals))
                            

    return eigenvalues, relevant_times