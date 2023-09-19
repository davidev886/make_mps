import numpy as np
import matplotlib.pyplot as plt
from src.utils import (generate_mps,
                       compute_mutual_info_matrix,
                       plot_mutual_info)


if __name__ == "__main__":
    graphics = True # True for plotting the matrix of the mutual information
    n_qubits = 20

    # create a GHZ state on n_qubits
    state_vector = np.zeros((2 ** n_qubits, 1), dtype=complex)
    state_vector[0,0] = 1
    state_vector[2 ** n_qubits - 1, 0] = 1
    normalization = np.sqrt(np.dot(state_vector.conj().T, state_vector))
    state_vector = state_vector / normalization

    # compute the MPS
    mps_psi = generate_mps(state_vector, cutoff=1.e-16)
    mutual_info_matrix = compute_mutual_info_matrix(mps_psi)

    if graphics:
        plot_mutual_info(mutual_info_matrix)
        plt.show()


