# import numba
import numpy as np

def fast_misorientation_angle(theta, phi=None):
    """
    
    """
    if phi is None: phi = np.zeros_like(theta)
        
    theta_radians = np.radians(theta)
    phi_radians = np.radians(phi)

    x = np.cos(theta_radians) * np.sin(phi_radians)
    y = np.sin(theta_radians) * np.sin(phi_radians)
    z = np.cos(phi_radians)

    cartesians = np.stack((x, y, z))
    cartesians /= np.linalg.norm(cartesians, axis=0)

    n_dims = len(theta.shape)
    is_3D = n_dims == 3

    # Padding
    if is_3D:
        padded = np.pad(cartesians, pad_width=[(0, 0), (1, 1), (1, 1), (1, 1)], mode='edge')
    else:
        padded = np.pad(cartesians, pad_width=[(0, 0), (1, 1), (1, 1)], mode='edge')

    # @numba.jit
    def disangle(padded: np.ndarray, axis: int):
        # Roll the array to align pixels with their first neighbour in the given axis, so as to be able to broadcast dot products
        padded_shifted = np.roll(padded, axis=axis+1, shift=1)
        # Compute dot products
        dot_prods = np.sum(np.reshape(padded, (len(padded), -1)) * np.reshape(padded_shifted, (len(padded_shifted), -1)), axis=0)
        # Reshape to image array shape (3D)
        dot_prods = np.reshape(dot_prods, padded.shape[1:])
        dot_prods = np.clip(dot_prods, -1, 1)
        misorientation_angle = np.arccos(dot_prods)
        misorientation_angle = np.degrees(misorientation_angle)
        misorientation_angle = misorientation_angle[None]
        
        # Resolve symmetry
        disangle_pos = np.min(np.concatenate((misorientation_angle, 180-misorientation_angle)), axis=0)

        # We get disorientation with the opposite neighbour in this axis as well, for free.
        disangle_neg = np.roll(disangle_pos, shift=1, axis=axis)

        disangle = np.max(np.stack((disangle_pos, disangle_neg)), axis=0)

        return disangle
    
    disangle_max = np.max(
        np.stack(
            (
                disangle(padded, axis=a)
                    for a in range(n_dims)
            )
        )
        , axis=0
    )

    # Remove the padding
    if is_3D:
        disangle_max = disangle_max[1:-1, 1:-1, 1:-1]
    else:
        disangle_max = disangle_max[1:-1, 1:-1]

    return disangle_max