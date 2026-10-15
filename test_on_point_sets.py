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

def main(method = "new_chains", n_times = 10, threshold = 0.3):
    if os.path.isfile(f"results/timings/unif3D_{method}_t={threshold}.csv"):
        results = pd.read_csv(f"results/timings/unif3D_{method}_t={threshold}.csv", index_col=0)
        results = results.to_dict(orient='index')
        print(results)
    else:
        results = {}

    for n_points in range(10, 95, 5):
        if n_points in results:
            print(f"Results for {n_points} points already exist. Skipping...")
            continue
        print(f"Running for {n_points} points...")
        random_points = np.loadtxt(f"point_sets/3D_uniform_{n_points}.csv", delimiter=",")
        print(random_points.shape)
        results[n_points] = {}

        if method == "new_chains":
            f = dio.fill_rips(random_points, 4, threshold)
            for time_i in range(n_times):
                stime = time.time()
                eigenvalues, relevant_times = complete_multiplicity_operator_eigenvalues(f=f, max_dim=3, up_only=False, fill_eigenvalues=True)
                results[n_points][str(time_i)] = time.time() - stime
        elif method == "petls":
            petls_complex = petls.Rips(points = random_points, threshold=threshold)
            request_list = petls_complex.filtration_list_to_spectra_request(petls_complex.get_all_filtrations(),dims=[0, 1, 2, 3], allpairs=True)
            for time_i in tqdm(range(n_times), desc="PETLS"):
                stime = time.time()
                spectra = petls_complex.eigenpairs(request_list=request_list)
                results[n_points][str(time_i)] = time.time() - stime
        elif method == "petls_combined":
            for time_i in tqdm(range(n_times)):
                stime = time.time()
                eigenvalues, relevant_times = petls_Rips_combined(random_points, threshold=threshold, max_dim=3, round_to=None, fill_eigenvalues=True)
                results[n_points][str(time_i)] = time.time() - stime
        elif method == "pinv":
            f = dio.fill_rips(random_points, 4, threshold)
            for time_i in range(n_times):
                stime = time.time()
                eigenvalues, relevant_times = complete_multiplicity_operator_eigenvalues(f=f, max_dim=3, up_only=False, fill_eigenvalues=False)
                results[n_points][str(time_i)] = time.time() - stime
        else:
            raise ValueError("Invalid method. Choose from 'new_chains', 'petls', or 'pinv'.")

    # Save results to a CSV file
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(f"results/timings/unif3D_{method}_t={threshold}.csv", index_label='n_points')




if __name__ == "__main__":
    main(method="new_chains", n_times=10, threshold=0.3)