import numpy as np
from numpy.linalg import norm, eig


def rotate(v, thetax, thetay, thetaz):
    # angles are given in degrees
    # vect should be 1x3 dimensions

    v_new = np.zeros(np.shape(v))
    thetax = thetax * np.pi / 180
    thetay = thetay * np.pi / 180
    thetaz = thetaz * np.pi / 180
    
    Rx = np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(thetax), -np.sin(thetax)],
            [0, np.sin(thetax), np.cos(thetax)],
        ]
    )
    Ry = np.matrix(
        [
            [np.cos(thetay), 0, np.sin(thetay)],
            [0, 1, 0],
            [-np.sin(thetay), 0, np.cos(thetay)],
        ]
    )
    Rz = np.matrix(
        [
            [np.cos(thetaz), -np.sin(thetaz), 0],
            [np.sin(thetaz), np.cos(thetaz), 0],
            [0, 0, 1],
        ]
    )

    R = Rx * Ry * Rz
    v_new += R * v

    return v_new


def transform(v1, v2, axis=None):
    # convert to coords of principal axis (v2)
    # Take transpose so that v1[0],v1[1],v1[2] are all x,y,z respectively
    v1 = v1.T
    v_new = np.zeros(np.shape(v1))

    # loop over each of the 3 coorinates
    if axis == None:
        for i in range(3):
            v_new[i] += v1[0] * v2[i, 0] + v1[1] * v2[i, 1] + v1[2] * v2[i, 2]
        return v_new
    else:
        v_new[0] += v1[0] * v2[axis, 0] + v1[1] * \
            v2[axis, 1] + v1[2] * v2[axis, 2]
        return v_new


def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.

    Arguments:
        x: vector or set of vectors with dimension (n, 3), where n is the
            number of vectors

    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x.

    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()

    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([np.cos(x2) * np.sqrt(x3), np.sin(x2)
                 * np.sqrt(x3), np.sqrt(1 - x3)])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)

    return ((x - mean_coord) @ M) + mean_coord @ M


def rotate_position(host_I, pos, rvir):
    """Transform particle position to randomly rotated coordinate system

    Args:
        host_I (_type_): host halo inertia tensor
        pos (_type_): particle position coordinates (x,y,z)
        rvir (_type_): halo virial radius

    Returns:
        _type_: 2d projected distance of particle from halo center
    """
    hw, hv = get_eigs(host_I, rvir)
    new_pos = pos

    new_hv = uniform_random_rotation(hv)

    hA = new_hv[0]
    hA2 = np.repeat(hA, len(new_pos)).reshape(3, len(new_pos)).T
    para1 = (new_pos * hA2 / norm(hA)).sum(axis=1)
    para2 = (hA / norm(hA)).T
    para = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))
    perp = new_pos - para.T
    r = np.sqrt(np.sum(perp**2, axis=1))

    angle = np.dot(new_hv[0], hv[0]) / (
        np.linalg.norm(new_hv[0]) * np.linalg.norm(hv[0])
    )  # -> cosine of the angle

    return new_hv, angle, hv


def get_eigs(I, rvir):
    # return eigenvectors and eigenvalues
    w, v = eig(I)
    # sort in descending order
    odr = np.argsort(-1.0 * w)
    # sqrt of e values = a,b,c
    w = np.sqrt(w[odr])
    v = v.T[odr]
    # rescale so major axis = radius of original host
    ratio = rvir / w[0]
    w[0] = w[0] * ratio  # this one is 'a'
    w[1] = w[1] * ratio  # b
    w[2] = w[2] * ratio  # c

    return w, v


def get_perp_dist(host_I, rvir, pos):
    hw, hv = get_eigs(host_I, rvir)

    new_pos = transform(pos, hv).T
    hA = host_I[0]
    hA2 = np.repeat(hA, len(new_pos)).reshape(3, len(new_pos)).T

    para1 = (new_pos * hA2 / norm(hA)).sum(axis=1)
    para2 = (hA / norm(hA)).T
    para = np.array((para2[0] * para1, para2[1] * para1, para2[2] * para1))
    perp = new_pos - para.T

    return perp
