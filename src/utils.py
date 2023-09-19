from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
import numpy as np
import matplotlib.pyplot as plt


def generate_mps(state_vector, cutoff=1.e-16):
    """
    creates an MPS from the state vector
    :param cutoff: float
        Cutoff of singular values used in the SVDs.
    :param state_vector: numpy array / list
        represents the state vector
    :return: mps object from tenpy
    """
    state_vector = np.array(state_vector)

    normalization = np.sqrt(np.dot(state_vector.conj().T, state_vector)).item()
    assert abs(normalization - 1) < 1e-8, "psi not normalized"

    L = int(np.log2(state_vector.size))
    # need to reshape the state_vector as a tensor with L physical indices
    state_vector_tensor = state_vector.reshape((2,) * L)
    psi = npc.Array.from_ndarray_trivial(state_vector_tensor, labels=[f"p{j}" for j in range(L)])

    site = SpinHalfSite(conserve=None)
    mps_A = MPS.from_full([site] * L, psi, cutoff=cutoff)
    return mps_A


def compute_mutual_info_matrix(mps_psi):
        """
        COmpute the mutual information I_{ij} where ij run on all the possible couples
        of lattice sites
        :param mps_psi: MPS
        :return: a matrix with the von neumann entropy on the diagonal and the
        mutual information on the off diagonal elements
        """
        mutual_info_list = mps_psi.mutinf_two_site()[1]
        entanglement_entropy = mps_psi.entanglement_entropy_segment()
        mutual_info_matrix = np.zeros((mps_psi.L, mps_psi.L))
        mutual_info_matrix[np.triu_indices(mps_psi.L, k=1)] = mutual_info_list
        mutual_info_matrix[np.tril_indices(mps_psi.L, k=-1)] = mutual_info_list
        mutual_info_matrix += np.diag(entanglement_entropy)

        return mutual_info_matrix


def plot_mutual_info(mutual_info_matrix):
    """
    plot the matrix of the mutual information

    :param mutual_info_matrix: numpy array
    """
    L = mutual_info_matrix.shape[0]
    ax = plt.subplot()
    ax.imshow(mutual_info_matrix, interpolation=None)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticks(np.arange(0, L, 1))
    ax.set_yticks(np.arange(0, L, 1))
    ax.set_xticks(np.arange(-.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-.5, L, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)

