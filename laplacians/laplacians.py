import array
from cProfile import label
from matplotlib.pylab import LinAlgError, lstsq
import numpy as np
from copy import deepcopy
import dionysus as dio
import scipy
import scipy.stats as ss
import pandas as pd
from sympy import use
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gudhi as gd

import time

from .general import time_to_idx, compute_boundary_matrices

################################################################
################ Combinatorial Laplacian #######################
################################################################

def combinatorial_Laplacian(q, t=None, t_i=None, 
                            boundary_matrices = None, simplices_at_time = None, f: dio.Filtration = None):
    """
    Computes the combinatorial Laplacian at time t.
    """
    assert (boundary_matrices is not None and simplices_at_time is not None) or f is not None, "Either boundary_matrices and simplices_at_time or filtration f must be provided"

    if f is not None:
        boundary_matrices, simplices_at_time, relevant_times = compute_boundary_matrices(f)

    if t_i is None:
        t_i = time_to_idx(t, relevant_times)

    Bq = boundary_matrices[q][:simplices_at_time[t_i][q-1], :simplices_at_time[t_i][q]]
    Bqp1 = boundary_matrices[q+1][:simplices_at_time[t_i][q], :simplices_at_time[t_i][q+1]]

    return Bqp1 @ Bqp1.T + Bq.T @ Bq 

    
def combinatorial_Laplacian_eigenvalues(boundary_matrices = None, simplices_at_time = None, relevant_times = None, f: dio.Filtration = None,
                                         weight_fun= lambda x: 1, max_dim=1):
    """
    Computes the eigenvalues of the combinatorial Laplacian at all times.
    """
    assert (boundary_matrices is not None and simplices_at_time is not None) or f is not None, "Either boundary_matrices and simplices_at_time or filtration f must be provided"
    if f is not None:
        boundary_matrices, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {}
    for q in range(max_dim+1):
        eigenvalues[q] = {}
        for t_i, t in tqdm(enumerate(relevant_times), leave=False, desc=f"Computing eigenvalues for q={q}"):
            L = combinatorial_Laplacian(q, t_i=t_i, boundary_matrices=boundary_matrices, simplices_at_time = simplices_at_time)
            if L.shape[0] == 0:
                eigenvalues[q][t] = {t: np.array([])}
                continue
            try:
                eigvals = np.linalg.eigvalsh(L)
                eigenvalues[q][t] = {t: eigvals}
            except LinAlgError:
                print(f"LinAlgError at time {t} for q={q}.")
                eigenvalues[q][t] = {t: np.array([])}

    return eigenvalues, relevant_times

################################################################
################### Persistent Laplacian #######################
################################################################

def schur_complement(M, m, n, method='compute_inverse'):
    """
    Compute the Schur complement of a matrix M.
    Function from "The Persistent Laplacian for Data Science: Evaluating Higher-Order Persistent  Spectral Representations of Data" by Thomas Davies, Zhengchao Wan and Ruben Sanchez-Garcia
    
    Args:
        M (np.ndarray): The input matrix.
        m (int): The index to split the matrix.
        n (int): The index to split the matrix.
        method (str): The method to use for computing the Schur complement. Options are 'compute_inverse' and 'least_squares'.
    """
    # require 0 < m < n

    if len(M.shape) != 2 and M.shape[0] != M.shape[1]:
        raise ValueError('Schur Complement error: matrix not square')

    if method == 'compute_inverse':
        inv = np.linalg.pinv(M[m:n, m:n])
        return M[0:m, 0:m] - M[0:m, m:n] @ inv @ M[m:n, 0:m]

    elif method == 'least_squares':
        # min (A*B) = A^{-1} * B
        inv_product = np.linalg.lstsq(M[m:n, m:n], M[m:n, 0:m], rcond=None)[0]
        # print(f"M0:\n{M[m:n, m:n]}\nM1:\n{M[m:n, 0:m]}\nlstq:\n{ np.linalg.lstsq(M[m:n, 0:m], M[m:n, m:n].T, rcond=None)}")
        return M[0:m, 0:m] - M[0:m, m:n] @ inv_product

    else:
        raise ValueError('Invalid Schur complement method')

def persistent_Laplacian_from_boundary(Bqplus1: np.ndarray, Bq: np.ndarray, verb = False, up_only=False, inverse_method = "compute_inverse") -> np.ndarray:
    """
    Inclusion of K in L. Assume the boundaries are ordered the same and first the boundaries in K appear in L.

    Args:
        Bqplus1 (np.ndarray): (q+1)-boundary matrix of L.
        Bq (np.ndarray): q-boundary matrix of K. (nqK if q=0).
        verb (bool): If True, print debug information.
        up_only (bool): If True, return only the up Laplacian.
    """
    nqK = Bq.shape[1]
    downL = Bq.T@Bq

    if Bqplus1 is not None:
        upLaplacian = Bqplus1@Bqplus1.T
        nqL = Bqplus1.shape[0]
        upL = schur_complement(upLaplacian, nqK, nqL, method=inverse_method)
    else:
        upL = 0

    if verb:
        # Print eigenvalues and eigenvectors
        print(f"upL:\n{upL}")
        if type(upL) != int:
            eigval, eigvec = np.linalg.eigh(upL)
            for i, val in enumerate(eigval):
                print(f"eval: {np.round(val,2)}, evec: {np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),decimals=2)}")
        print()
        print(f"downL:\n{downL}")
        if type(downL) != int:
            eigval, eigvec = np.linalg.eigh(downL)
            for i, val in enumerate(eigval):
                print(f"eval: {np.round(val,2)}, evec: {np.round(eigvec[:,i]/np.min(np.abs(eigvec[:,i][np.abs(eigvec[:,i]) > 1e-8])),decimals=2)}")
        print()

    if up_only:
        return upL
    return upL + downL

def persistent_Laplacian(q, s=None, t=None, s_i=None, t_i=None,
                         boundary_matrices = None, simplices_at_time = None, relevant_times = None, f: dio.Filtration = None, 
                         verb=False, up_only = False, inverse_method = "compute_inverse") -> np.ndarray:
    """
    Compute the persistent Laplacian at dimension q between filtration steps s and t.

    Args:
        q (int): Dimension.
        s (int): Filtration step for K.
        t (int): Filtration step for L.
        boundary_matrices (list of np.ndarray, optional): List of boundary matrices for each dimension.
        simplices_at_time (function, optional): Function that returns the number of simplices at each dimension for a given filtration step.
        f (dio.Filtration, optional): Filtration from Dionysus to compute boundary matrices if not provided.
        verb (bool): If True, print debug information.
        up_only (bool): If True, return only the up Laplacian
    """

    assert (boundary_matrices is not None and simplices_at_time is not None) or f is not None, "Either boundary_matrices and simplices_at_time or filtration f must be provided"

    if f is not None:
        boundary_matrices, simplices_at_time, relevant_times = compute_boundary_matrices(f)

    if len(boundary_matrices) <= q:
        return None

    if s_i is None:
        s_i = time_to_idx(s, relevant_times)
    if t_i is None:
        t_i = time_to_idx(t, relevant_times)
    
    if q > 0:
        Bq = boundary_matrices[q][:simplices_at_time[s_i][q-1], :simplices_at_time[s_i][q]]
    else:
        Bq = np.zeros((simplices_at_time[s_i][q], simplices_at_time[s_i][q]))
    
    if q+1 < len(boundary_matrices):
        Bqplus1 = boundary_matrices[q+1][:simplices_at_time[t_i][q], :simplices_at_time[t_i][q+1]]
    else:
        Bqplus1 = None

    if verb:
        print(f"Bqplus1:\n{Bqplus1}")
    return persistent_Laplacian_from_boundary(Bqplus1, Bq, verb=verb, up_only=up_only, inverse_method=inverse_method)

def persistent_Laplacian_eigenvalues(boundary_matrices = None, simplices_at_time = None, f: dio.Filtration = None,
                                     weight_fun = lambda x: 1, max_dim = 1, up_only=False, inverse_method = "compute_inverse"):
    assert (boundary_matrices is not None and simplices_at_time is not None) or f is not None, "Either boundary_matrices and simplices_at_time or filtration f must be provided"

    if f is not None:
        boundary_matrices, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False, desc=f"Computing eigenvalues for q={q}")
        for t_i in t_i_bar:
            for s_i in range(t_i+1):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                Lap = persistent_Laplacian(q, s_i=s_i, t_i=t_i, boundary_matrices=boundary_matrices, simplices_at_time=simplices_at_time, up_only=up_only, inverse_method=inverse_method)

                if Lap is not None:
                    eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap)
    return eigenvalues, relevant_times

def horizontal_eigenvalues(boundary_matrices = None, simplices_at_time = None, f: dio.Filtration = None,
                                     weight_fun = lambda x: 1, max_dim = 1, up_only=False, inverse_method = "compute_inverse"):
    assert (boundary_matrices is not None and simplices_at_time is not None) or f is not None, "Either boundary_matrices and simplices_at_time or filtration f must be provided"

    if f is not None:
        boundary_matrices, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    for q in range(max_dim+1):
        s_i_bar = tqdm(range(len(relevant_times)), leave=False, desc=f"Computing eigenvalues for q={q}")
        for s_i in s_i_bar:
            prev_Lap = None
            for t_i in range(s_i, len(relevant_times)):
                s_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                Lap = persistent_Laplacian(q, s_i=s_i, t_i=t_i, boundary_matrices=boundary_matrices, simplices_at_time=simplices_at_time, up_only=up_only, inverse_method=inverse_method)
                if prev_Lap is not None:
                    eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap-prev_Lap)
                else:
                    eigenvalues[q][s][t] = np.array([])#np.linalg.eigvalsh(Lap)
                prev_Lap = deepcopy(Lap)
    return eigenvalues, relevant_times

def vertical_eigenvalues(boundary_matrices = None, simplices_at_time = None, f: dio.Filtration = None,
                                     weight_fun = lambda x: 1, max_dim = 1, up_only=False, inverse_method = "compute_inverse"):
    assert (boundary_matrices is not None and simplices_at_time is not None) or f is not None, "Either boundary_matrices and simplices_at_time or filtration f must be provided"

    if f is not None:
        boundary_matrices, simplices_at_time, relevant_times = compute_boundary_matrices(f, weight_fun)

    eigenvalues = {q: {s: {t: np.array([]) for t in relevant_times} for s in relevant_times} for q in range(max_dim+1)}
    
    for q in range(max_dim+1):
        t_i_bar = tqdm(range(len(relevant_times)), leave=False, desc=f"Computing eigenvalues for q={q}")
        for t_i in t_i_bar:
            prev_Lap = None
            for s_i in range(t_i+1):
                t_i_bar.set_description(f"s_i: {s_i}/{t_i}")
                s, t = relevant_times[s_i], relevant_times[t_i]

                Lap = persistent_Laplacian(q, s_i=s_i, t_i=t_i, boundary_matrices=boundary_matrices, simplices_at_time=simplices_at_time, up_only=up_only, inverse_method=inverse_method)
                if prev_Lap is not None:
                    prev_Lap = np.pad(prev_Lap, ((0, Lap.shape[0]-prev_Lap.shape[0]), (0, Lap.shape[1]-prev_Lap.shape[1])), mode='constant', constant_values=0)
                    eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap-prev_Lap)
                else:
                    eigenvalues[q][s][t] = np.linalg.eigvalsh(Lap)
                prev_Lap = deepcopy(Lap)
    return eigenvalues, relevant_times
