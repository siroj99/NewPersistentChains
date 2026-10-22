# driver.py
import subprocess
import sys

n_times = 5

for seed_i in range(25):
    for method in ["new_chains", "petls", "petls_combined", "pinv"]:
        for point_set in ["unif2D", "unif3D", "sphere3D", "unif100D"]:
            if point_set in ["unif2D", "sphere3D"]:
                filtration_type = "alpha"
                if point_set == "unif2D":
                    min_points = 10
                    if method in ["new_chains", "petls_combined"]:
                        max_points = 150
                    else:
                        max_points = 80
                    step_size = 10
                    seed = seed_i
                elif point_set == "sphere3D":
                    min_points = 10
                    if method in ["new_chains", "petls_combined"]:
                        max_points = 150
                        step_size = 10
                    else:
                        max_points = 70
                        step_size = 10
                    seed = seed_i + 2000
            else:
                filtration_type = "rips"
                if point_set == "unif100D":
                    threshold = 5
                    min_points = 5
                    max_points = 15
                    step_size = 1
                    seed = seed_i + 3000
                elif point_set == "unif3D":
                    threshold = 0.3
                    min_points = 10
                    max_points = 90
                    step_size = 5
                    seed = seed_i + 1000

            for n_points in range(min_points, max_points+1, step_size):            
                args = [sys.executable, "test_on_point_sets.py", "--method", method, "--point_set", point_set,
                        "--seed", str(seed), "--filtration_type", filtration_type, "--n_points", str(n_points), "--n_times", str(n_times)]
                if filtration_type == "rips":
                    args += ["--threshold", str(threshold)]
                subprocess.run(args)