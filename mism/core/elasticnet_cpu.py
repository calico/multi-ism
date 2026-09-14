import numpy as np
import multiprocessing
from celer import ElasticNet
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack

# Global variables for multiprocessing
X = None
Y = None
coefs = None


def save_sparse_matrix_to_hdf5(h5_group, matrix_name, sparse_matrix):
    """
    Save a sparse matrix to an HDF5 group.

    Args:
        h5_group: HDF5 group object
        matrix_name: Name for the matrix group
        sparse_matrix: scipy.sparse matrix to save
    """
    # Create a group for the sparse matrix
    matrix_group = h5_group.create_group(matrix_name)

    # Save the sparse matrix components
    matrix_group.create_dataset("data", data=sparse_matrix.data, compression="gzip")
    matrix_group.create_dataset(
        "indices", data=sparse_matrix.indices, compression="gzip"
    )
    matrix_group.create_dataset("indptr", data=sparse_matrix.indptr, compression="gzip")

    # Save metadata as attributes
    matrix_group.attrs["shape"] = sparse_matrix.shape
    matrix_group.attrs["format"] = sparse_matrix.format
    matrix_group.attrs["dtype"] = str(sparse_matrix.dtype)

    print(
        f"Saved sparse matrix '{matrix_name}' with shape {sparse_matrix.shape} and {sparse_matrix.nnz:,} non-zero elements"
    )


def load_sparse_matrix_from_hdf5(h5_group, matrix_name):
    """
    Load a sparse matrix from an HDF5 group.

    Args:
        h5_group: HDF5 group object
        matrix_name: Name of the matrix group

    Returns:
        scipy.sparse matrix
    """
    matrix_group = h5_group[matrix_name]

    # Load the sparse matrix components
    data = matrix_group["data"][:]
    indices = matrix_group["indices"][:]
    indptr = matrix_group["indptr"][:]

    # Load metadata
    shape = tuple(matrix_group.attrs["shape"])
    matrix_format = matrix_group.attrs["format"]
    dtype = matrix_group.attrs["dtype"]

    # Reconstruct the sparse matrix
    if matrix_format == "csr":
        sparse_matrix = csr_matrix((data, indices, indptr), shape=shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported sparse matrix format: {matrix_format}")

    print(
        f"Loaded sparse matrix '{matrix_name}' with shape {shape} and {sparse_matrix.nnz:,} non-zero elements"
    )
    return sparse_matrix


def fit_elasticnet(args):
    i, alpha = args
    y = Y[:, i]
    if coefs is not None:
        clr = ElasticNet(
            alpha=alpha,
            l1_ratio=1,
            fit_intercept=False,
            max_iter=500,
            warm_start=True,
            tol=1e-3,
            max_epochs=10000,
            verbose=0,
        )
        clr.coef_ = coefs[:, i].astype(
            X.dtype
        )  # Match X's dtype to avoid buffer mismatch
        clr.fit(X, y)
    else:
        clr = ElasticNet(
            alpha=alpha,
            l1_ratio=1,
            fit_intercept=False,
            max_iter=500,
            tol=1e-3,
            max_epochs=10000,
            verbose=0,
        )
        clr.fit(X, y)
    return clr.coef_


def ElasticNetMulti(
    X_input, Y_input, njobs, alpha, coefs_input=None, return_sparse=False
):
    """
    Run Lasso regression in parallel across Y's columns with progress tracking.

    Parameters:
    - X_input: feature matrix (n_samples, n_features)
    - Y_input: target matrix (n_samples, n_tasks)
    - njobs: number of parallel jobs
    - alpha: regularization parameter
    - coefs_input: initial coefficients matrix (n_features, n_tasks), default: None
        If provided, will be used for warm start initialization
    - return_sparse: if True, sparsify each task's coefficient vector as it is
        collected and return a scipy CSR matrix, avoiding the dense (n_tasks,
        n_features) stack (which dominates peak memory for large, sparse Lasso
        solutions). Default False returns the dense array.

    Returns:
    - coef_matrix: (n_tasks, n_features); dense ndarray, or CSR if return_sparse.
    """

    # Set global variables for worker processes
    global X, Y, coefs
    X = X_input
    Y = Y_input
    coefs = coefs_input

    # Required for Unix-like systems to avoid RuntimeError
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    fit_func = fit_elasticnet
    desc = "L1 fitting"

    print(f"Starting {desc} for {Y.shape[1]} targets using {njobs} processes...")

    # Create argument tuples for each target
    args_list = [(i, alpha) for i in range(Y.shape[1])]

    with multiprocessing.Pool(processes=njobs) as pool:
        # Use imap with tqdm for progress tracking
        result_iter = tqdm(
            pool.imap(fit_func, args_list), total=Y.shape[1], desc=desc, unit="targets"
        )
        if return_sparse:
            # sparsify each coef vector as it arrives so the full dense
            # (n_tasks, n_features) stack is never materialized
            rows = [csr_matrix(coef) for coef in result_iter]
            return vstack(rows, format="csr")
        results = list(result_iter)

    return np.array(results)
