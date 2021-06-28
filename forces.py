import numpy as np

def cartesian_to_spherical(pos: np.ndarray) -> np.ndarray:
    theta_phi = np.empty(shape=pos.shape[:-1] + (3, ))

    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)
    theta_phi[..., 0] = r
    theta_phi[..., 1] = np.arccos(z / r)  # theta
    theta_phi[..., 2] = np.arctan2(y, x)  # phi

    return theta_phi

def spherical_to_cartesian(theta_phi: np.ndarray) -> np.ndarray:
    theta, phi = theta_phi[..., 0], theta_phi[..., 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)

methane = np.array([[-0.02209687,  0.00321505,  0.01651974],
                   [-0.66900878,  0.88935986, -0.1009085 ],
                   [-0.37778794, -0.85775189, -0.58829603],
                   [ 0.09642092, -0.3151253 ,  1.06378087],
                   [ 0.97247267,  0.28030227, -0.39109608]])

print(cartesian_to_spherical(methane[0]))
print(cartesian_to_spherical(methane[1]))
print(cartesian_to_spherical(methane))
print(spherical_to_cartesian(coords[:,1:]))