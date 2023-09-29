# import numba
import numpy as np

def fast_misorientation_angle(theta, phi=None):
    """
    Misorientation angle calculation in 2D and 3D orientation matrices.
    """
    if phi is None: phi = np.zeros_like(theta)
        
    n_dims = len(theta.shape)
    is_3D = n_dims == 3

    theta_radians = np.radians(theta)
    phi_radians = np.radians(phi)

    if is_3D:
        x = np.cos(theta_radians) * np.sin(phi_radians)
        y = np.sin(theta_radians) * np.sin(phi_radians)
        z = np.cos(phi_radians)
        cartesians = np.stack((x, y, z))
        cartesians /= np.linalg.norm(cartesians, axis=0)
    else:
        x = np.cos(theta_radians)
        y = np.sin(theta_radians)
        cartesians = np.stack((x, y))
        cartesians /= np.linalg.norm(cartesians, axis=0)

    # @numba.jit
    def disangle(cartesians: np.ndarray, axis: int):
        # Roll the array to align pixels with their first neighbour in the given axis
        shifted = np.roll(cartesians, axis=axis+1, shift=1)
        
        if is_3D:
            shifted = shifted[:, 1:-1, 1:-1, 1:-1]
            padded_shifted = np.pad(shifted, pad_width=[(0, 0), (1, 1), (1, 1), (1, 1)], mode='edge')
        else:
            shifted = shifted[:, 1:-1, 1:-1]
            padded_shifted = np.pad(shifted, pad_width=[(0, 0), (1, 1), (1, 1)], mode='edge')

        dot_prods = np.sum(np.reshape(cartesians, (len(cartesians), -1)) * np.reshape(padded_shifted, (len(padded_shifted), -1)), axis=0)
        dot_prods = np.reshape(dot_prods, cartesians.shape[1:])
        dot_prods = np.clip(dot_prods, -1, 1)
        misorientation_angle = np.arccos(dot_prods)
        misorientation_angle = np.degrees(misorientation_angle)
        misorientation_angle = misorientation_angle[None]
        
        # Resolve symmetry
        disangle_pos = np.min(np.concatenate((misorientation_angle, 180-misorientation_angle)), axis=0)

        # Disorientation with the opposite pixel neighbour
        disangle_neg = np.roll(disangle_pos, shift=-1, axis=axis)
        disangle = np.max(np.stack((disangle_pos, disangle_neg)), axis=0)

        return disangle
    
    disangle_max = np.max(
        np.stack(
            (
                disangle(cartesians, axis=a)
                    for a in range(n_dims)
            )
        )
        , axis=0
    )

    return disangle_max