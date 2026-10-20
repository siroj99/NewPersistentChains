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

def main(method = "new_chains", point_set = "unif3D_0", n_times = 10, threshold = 0.3, filtration_type = "rips"):
    if filtration_type == "rips":
        filename = f"{point_set}_{method}_t={threshold}.csv"
    else:
        filename = f"{point_set}_{method}.csv"
    if os.path.isfile(f"results/timings/{filtration_type}/{filename}"):
        results = pd.read_csv(f"results/timings/{filtration_type}/{filename}", index_col=0)
        results = results.to_dict(orient='index')
        print(results)
    else:
        results = {}

    point_files = os.listdir(f"point_sets/{point_set}")
    for i, point_file in enumerate(point_files):
        n_points = int(point_file.split("=")[1].split(".")[0])
        point_files[i] = (n_points, point_file)
    point_files.sort(key=lambda x: x[0])

    for n_points, point_file in point_files:
        if not point_file.endswith(".csv"):
            continue
        if n_points in results:
            print(f"Results for {n_points} points already exist. Skipping...")
            continue
        
        print(f"Running for {n_points} points...")
        random_points = np.loadtxt(f"point_sets/{point_set}/{point_file}", delimiter=",")
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
            elif filtration_type == "alpha":
                petls_complex = petls.Alpha(points = random_points, max_dim=4)
            else:
                raise ValueError("Invalid filtration type. Choose from 'rips' or 'alpha'.")
            request_list = petls_complex.filtration_list_to_spectra_request(petls_complex.get_all_filtrations(),dims=[0, 1, 2, 3], allpairs=True)

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
                petls_complex = petls.Alpha(points = random_points, max_dim=4)
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
        if not os.path.exists(f"results/timings/{filtration_type}"):
            os.makedirs(f"results/timings/{filtration_type}")
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(f"results/timings/{filtration_type}/{filename}", index_label='n_points')




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run timing experiments for different methods.")
    parser.add_argument("--method", type=str, default="new_chains", help="Method to use: 'new_chains', 'petls', 'petls_combined', or 'pinv'")
    parser.add_argument("--point_set", type=str, default="unif3D_0", help="Point set to use, e.g., 'unif3D_0'")
    parser.add_argument("--n_times", type=int, default=10, help="Number of times to run each experiment for averaging")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for Rips complex")
    parser.add_argument("--filtration_type", type=str, default="rips", help="Filtration type: 'rips' or 'alpha'")
    args = parser.parse_args()

    main(method=args.method, point_set=args.point_set, n_times=args.n_times, threshold=args.threshold, filtration_type=args.filtration_type)