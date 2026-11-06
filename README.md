The repository is split into 3 different notebooks.
- laplacians_test:      
  - For examples on how to use the introduced functions.
- generating_test_cases:               
  - Code to create the test cases used in the paper.
- analyzing_test_cases:                      
  - Notebook that makes the figures in the paper.

Most of the functions are in the directory `laplacians`. Most of the calculations require a dionysus filtration. To allow for more variety, we include a function `gudhi_to_dio` in the `general.py` that converts a gudhi simplex tree to a dionysus filtration. To compute persistent Laplacians, we require boundary matrices, which are calculated either automatically or manually by calling `compute_boundary_matrices`. This also return `simplices_at_time` and `relevant_times`, which are needed in a lot of downstream methods.

In the same directory, `laplacians.py` contains most basic functions for calculating the combinatorial Laplacian, persistent Laplacian using Schur complement and the horizontal and vertical operators also using the Schur complement. These functions return a dictionary `eigenvalues` that contains the dimension, start and end times with the eigenvalues of the corresponding Laplacian.

To visualize the result, one can use the functions from `plot_operator.py`. Here `plot_combinatorial_eigenvalues` plots the combinatorial Laplacian eigenvalues, calculated by the previous method, as a 1D function. `plot_eigenvalues` plots the eigenvalues of the persistent Laplacian or the horizontal or vertical operators in a colormesh like is done in the paper. To visualize a dionysus filtration that does not contain many simplices, we include a function `draw_filtration` in `plot_complex.py`. Calling this latter function with a great amount of simplices will be very computationally expensive.

Our introduced method, new persistent chains, can be found in the file `new_persistent_chains.py`. The function `new_persistent_chains` calculates the same eigenvalues given by the function in `laplacians.py`, but now using the new algorithm. Given that the output is the same, they can be again plotted in the same way with `plot_eigenvalues`.

To compare with other methods, in `other_libraries.py` we give a function to rewrite the PETLS output to the format we use for plotting. Furthermore, we give a function that computes the relevant locations used in the new persistent chains method and evaluates PETLS on those locations. NOTE that this function does not yet give the correct output. PETLS returns the sum of the up- and down-Laplacian, which means that is hard to fill in other values as we cannot easily split the effect of up and down at every location. It is our understanding that this should be possible however with a slight alteration of either the PETLS method or the code stated here. Because it computes the relevant Laplacians, it is possible to use in the analysis for computational time in `test_on_point_sets.py`.

Finally, to run the tests for time taken from the same starting point with all variables cleared, we require an extra file `run_tests.py` that initiates all tests in order.