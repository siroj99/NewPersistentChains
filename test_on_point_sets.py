from laplacians import *

import dionysus as dio
import numpy as np
import time
import gudhi as gd
import matplotlib.pyplot as plt
import petls
import pandas as pd
from tqdm.auto import tqdm
import os
import argparse
import sys

def main(method = "new_chains", point_set = "unif3D", seed=0,  n_times = 10, threshold = 0.3, filtration_type = "rips", n_points=10):
    if filtration_type == "rips":
        filename = f"{point_set}_{seed}_t={threshold}.csv"
    else:
        filename = f"{point_set}_{seed}.csv"
        
    os.makedirs(f"results/timings/{filtration_type}/{method}", exist_ok=True)

    if os.path.isfile(f"results/timings/{filtration_type}/{method}/{filename}"):
        results = pd.read_csv(f"results/timings/{filtration_type}/{method}/{filename}", index_col=0)
        results = results.to_dict(orient='index')
    else:
        results = {}

    if n_points in results:
        print(f"Results for {n_points} points already exist. Skipping...")
        return

    total_points = np.loadtxt(f"point_sets/{point_set}/seed={seed}.csv", delimiter=",")
    
    print(f"Running for {n_points} points...")
    random_points = total_points[:n_points]
    print(random_points.shape)
    results[n_points] = {}

    if method == "new_chains":
        if filtration_type == "rips":
            f = dio.fill_rips(random_points, 4, threshold)
        elif filtration_type == "alpha":
            st = gd.AlphaComplex(points=random_points).create_simplex_tree()
            f = gudhi_to_dio(st)

        for time_i in range(n_times):
            stime = time.time()
            eigenvalues, relevant_times = complete_multiplicity_operator_eigenvalues(f=f, max_dim=3, up_only=False, fill_eigenvalues=True)
            results[n_points][str(time_i)] = time.time() - stime


    elif method == "petls":
        if filtration_type == "rips":
            petls_complex = petls.Rips(points = random_points, threshold=threshold)
            dims = [0, 1, 2, 3]
        elif filtration_type == "alpha":
            try:
                petls_complex = petls.Alpha(points = random_points, max_dim=4)
                dims = [0, 1, 2, 3]
            except:
                st = gd.AlphaComplex(points=random_points).create_simplex_tree()
                petls_complex = petls.Complex(simplex_tree=st)
                dims = [0, 1, 2]
        else:
            raise ValueError("Invalid filtration type. Choose from 'rips' or 'alpha'.")
        request_list = petls_complex.filtration_list_to_spectra_request(petls_complex.get_all_filtrations(),dims=dims, allpairs=True)

        for time_i in tqdm(range(n_times), desc="PETLS"):
            stime = time.time()
            spectra = petls_complex.eigenpairs(request_list=request_list)
            results[n_points][str(time_i)] = time.time() - stime


    elif method == "petls_combined":
        if filtration_type == "rips":
            f = dio.fill_rips(random_points, 4, threshold)
            petls_complex = petls.Rips(points = random_points, threshold=threshold)
        elif filtration_type == "alpha":
            st = gd.AlphaComplex(points=random_points).create_simplex_tree()
            f = gudhi_to_dio(st)
            try:
                petls_complex = petls.Alpha(points = random_points, max_dim=4)
            except:
                petls_complex = petls.Complex(simplex_tree=st)
        else:
            raise ValueError("Invalid filtration type. Choose from 'rips' or 'alpha'.")
        
        for time_i in tqdm(range(n_times)):
            stime = time.time()
            eigenvalues, relevant_times = petls_combined(random_points, f=f, threshold=threshold, petls_complex=petls_complex, max_dim=3, round_to=None, fill_eigenvalues=True, filtration_type=filtration_type)
            results[n_points][str(time_i)] = time.time() - stime


    elif method == "pinv":
        if filtration_type == "rips":
            f = dio.fill_rips(random_points, 4, threshold)
        elif filtration_type == "alpha":
            st = gd.AlphaComplex(points=random_points).create_simplex_tree()
            f = gudhi_to_dio(st)
        else:
            raise ValueError("Invalid filtration type. Choose from 'rips' or 'alpha'.")

        for time_i in range(n_times):
            stime = time.time()
            eigenvalues, relevant_times = persistent_Laplacian_eigenvalues(f=f, max_dim=3, up_only=False, inverse_method="least_squares")
            results[n_points][str(time_i)] = time.time() - stime
    else:
        raise ValueError("Invalid method. Choose from 'new_chains', 'petls', or 'pinv'.")

    # Save results to a CSV file every time we finish a point set to avoid losing data in case of interruptions
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(f"results/timings/{filtration_type}/{method}/{filename}", index_label='n_points')




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run timing experiments for different methods.")
    parser.add_argument("--method", type=str, default="new_chains", help="Method to use: 'new_chains', 'petls', 'petls_combined', or 'pinv'")
    parser.add_argument("--point_set", type=str, default="unif3D", help="Point set to use, e.g., 'unif3D', 'unif100D', etc.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation")
    parser.add_argument("--n_times", type=int, default=10, help="Number of times to run each experiment for averaging")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for Rips complex")
    parser.add_argument("--filtration_type", type=str, default="rips", help="Filtration type: 'rips' or 'alpha'")
    parser.add_argument("--n_points", type=int, default=10, help="Number of points to test")
    args = parser.parse_args()

    main(method=args.method, point_set=args.point_set, seed=args.seed, n_times=args.n_times, threshold=args.threshold, filtration_type=args.filtration_type, n_points=args.n_points)