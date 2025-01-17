import numpy as np
from numpy.linalg import norm, eig
import sys


class HaloOrientation():

    @staticmethod
    def rotate(v, phi, theta):
        """Rotate vectors in three dimensions

            Arguments:
                v: vector or set of vectors with dimension (3,n), where n is the number of axis
                phi: angle between 0 and 2pi
                theta: angle between 0 and pi

            Returns:
                Rotated vector or set of vectors with dimension (3,n) where n is the
                number of vectors
            """

        v_new = np.zeros(np.shape(v))

        Rz = np.matrix(
            [
                [1, 0, 0],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi), np.cos(phi)],
            ]
        )

        Rx = np.matrix(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        R = Rx * Rz
        v_new += R * v

        return v_new

    @staticmethod
    def transform(v1, v2):
        """transform to different coordinate system

        Arguments:
            v1: vector or set of vectors with dimension (3, n), where n is the
                number of vectors
            v2: principal axis of desired coordinate system

        Returns:
            Vector or set of vectors in new coordinate system with dimension
            (n, 3), where n is the number of vectors
        """

        v_new = np.zeros(np.shape(v1))

        # loop over each of thse 3 coordinates
        for i in range(3):
            v_new[i] += v1[0] * v2[i, 0] + v1[1] * v2[i, 1] + v1[2] * v2[i, 2]

        return v_new

    @staticmethod
    def uniform_rotation(v, phi=None, theta=None):
        """Do uniform random 3D rotation

        Args:
            v (array): (3, n) coordinate vector where n is number of vectors
            phi (float): angle in radians (0 -> 2pi) (default none for random angle)
            theta (float): angle in radians (0 -> pi) (default none for random angle)

        Returns:
            v_new (array): rotated vectors (3,n)
            phi (float): angle in radians (0 -> 2pi)
            theta (float): angle in radians (0 -> pi)
        """

        if phi == None:
            phi = 2*np.pi*np.random.random()

        if theta == None:
            theta = np.arccos(np.random.random())

        v_new = HaloOrientation.rotate(v, phi, theta)

        return v_new.T, phi, theta

    @staticmethod
    def get_eigs(I, rvir):
        """Get eigenvalues and eigenvectors of halo inertia tensor

        Arguments:
            I (array): host halo inertia tensor
            rvir (array): halo virial radius

        Returns:
            array: eigenvalues
            array: eigenvectors
        """
        # return eigenvectors and eigenvalues
        w, v = eig(I)

        # sort in descending order
        odr = np.argsort(-1.0 * w)

        # sqrt of e values = A,B,C where A is the major axis
        w = np.sqrt(w[odr])
        v = v.T[odr]

        # rescale so major axis = radius of original host
        ratio = rvir / w[0]
        w[0] = w[0] * ratio  # this one is 'A'
        w[1] = w[1] * ratio  # B
        w[2] = w[2] * ratio  # C

        return w, v

    @staticmethod
    def get_random_axes_and_angles(princip_axes, phi=None, theta=None):
        """Define set of randomly oriented axes by rotating principal axes
            and calculate azimuthal angle between new axis and major axis

        Args:
            princip_axes (array): Principal axes 3x3 array
                princip_axes[0] corresponds to major axis
                princip_axes.T[0] corresponds to i-th component of vectors
            phi (float): angle in radians (0 -> 2pi)
            theta (float): angle in radians (0 -> pi)

        Returns:
            new_axes (2d array): set of 3 orthogonal vectors
                shape is 3x3
                new_axes[0] corresponds to first vector
                new_axes.T[0] corresponds to i-th component of vectors

            angle (float): azimuthal angle between long axes of both axes sets
            phi (float): angle in radians (0 -> 2pi)
            theta (float): angle in radians (0 -> pi)
        """

        # returns angles in radians
        new_axes, phi, theta = HaloOrientation.uniform_rotation(
            princip_axes, phi, theta)

        hA = new_axes.T[0]

        angle = np.dot(hA, princip_axes[0]) / (
            norm(hA) * norm(princip_axes[0])
        )  # -> cosine of the angle

        return new_axes, angle, phi, theta

    @staticmethod
    def check_ortho(e_vect):
        """Check if eigenvectors inertia tensor are orthogonal

        Arguments:
            e_vect (array): 3x3 array of inertia tensor eigenvectors
                For consistency with rest of methods:
                    e_vect.T[0] corresponds to major axis
                    e_vect[0] corresponds to i-th component of vectors


        """

        # define a diagonal matrix of ones
        a = np.zeros((3, 3))
        np.fill_diagonal(a, 1.)

        # take dot product of e_vect and e_vect.T
        # off diagonals are usually 1e-15 so round them to 0.
        m = np.abs(np.round(np.dot(e_vect, e_vect.T), 1))

        # check if all elements are equal to identity matrix
        if np.any(a != m):
            sys.exit(1)

    @staticmethod
    def get_axis_ratios(e_values):
        """Return axis ratios from array of eigenvalues

        Arguments:
            e_values (array): 3x1 array of inertia tensor eigenvalues

        Returns:
            s: ratio of shortest axis to longest axis
            q: ratio of intermediate axis to longest axis
        """
        a, b, c = e_values
        s = c/a
        q = b/a

        return s, q

    @staticmethod
    def get_perp_dist(v, pos):
        """ Return component of vector perpendicular to major axis
        and its angular separation from major axis

        Arguments:
            v: vector defining axis of interest (example: major axis)
            pos: position (x,y,z) vector or set of vectors with dimension (n, 3), 
                where n is the number of vectors

        Returns:
            perp: component perpendicular to axis of interest
            angle: angular separation between axis of interest and position vector
        """

        v2 = np.repeat(v, len(pos)).reshape(3, len(pos)).T

        # dot product to find magnitude of component
        # of position vectors parallel to major axis
        v_dot = (pos * v2 / norm(v)).sum(axis=1)

        # normalized major axis vector
        v_hat = (v / norm(v)).T

        # parallel vector (magnitude and direction)
        para = np.array((v_hat[0] * v_dot, v_hat[1] * v_dot, v_hat[2] * v_dot))

        # perpendicular component
        perp = pos - para.T

        return perp

    @staticmethod
    def get_2d_shape(pos):
        """
        Get shape of halo projected in 2 dimensions and its position angle

        Args:
            pos (array): position (x,y,z) vector or set of vectors with dimension (n, 3), 
                where n is the number of vectors

        Returns:
            long_eig (float): length of longest axis.
            short_eig (float): length of shortest axis.

        """

        r2 = pos[0] ** 2 + pos[1] ** 2
        Ixx = np.sum((pos[1] * pos[1]) / r2)
        Iyy = np.sum((pos[0] * pos[0]) / r2)
        Ixy = np.sum((pos[0] * pos[1]) / r2)
        Iyx = Ixy

        I = np.array(((Ixx, -Ixy), (-Iyx, Iyy)))

        # return eigenvectors and eigenvalues
        w, v = eig(I)

        # sort in descending order
        odr = np.argsort(-1.0 * w)

        # sqrt of e values = a,b,c
        w = np.sqrt(w[odr])
        short_eig = w[1]
        long_eig = w[0]

        return long_eig, short_eig
    

    @staticmethod
    def get_projected_ellipse(C, B, A, th, ph):
        """
        https://phys.libretexts.org/Bookshelves/Astronomy__Cosmology/Celestial_Mechanics_(Tatum)/04%3A_Coordinate_Geometry_in_Three_Dimensions/4.03%3A_The_Ellipsoid
        
        Get the 2d projection of a triaxial ellipsoid with axis lengths A>B>C and
        coordinate axes x, y, z. Transform to a set of coordinate axes x',y',z'
        such that x' is in the direction (θ,𝜙), first by a rotation through
        𝜙 about x to form intermediate axes x1, y1, z1, followed by a rotation
        through θ about y1. 

        Args:
            A (float): length of minor axis
            B (float): lenght of semi-major axis
            C (float): length of major axis
            th (float): theta (0 to 90 rotates about semi-major axis)
            ph (float): phi (0 to 360 rotates about major axis)

        Returns:
            ax1 (float): new x axis length
            ax2 (float): new y axis length
            PS (float): position angle

        """
        A2 = A*A
        B2 = B*B
        C2 = C*C

        TH = th/57.29578
        PH = ph/57.29578

        STH = np.sin(TH)
        CTH = np.cos(TH)
        SPH = np.sin(PH)
        CPH = np.cos(PH)

        STH2 = STH*STH
        CTH2 = CTH*CTH
        SPH2 = SPH*SPH
        CPH2 = CPH*CPH

        AA = CTH2*(CPH2/A2+SPH2/B2)+STH2/C2
        TWOHH = 2.*CTH*SPH*CPH*((1./B2)-(1./A2))
        BB = SPH2/A2+CPH2/B2

        PS = .5*np.arctan2(TWOHH, AA-BB)
        SPS = np.sin(PS)
        CPS = np.cos(PS)
        AAA = CPS*(AA*CPS+TWOHH*SPS)+BB*SPS*SPS
        BBB = SPS*(AA*SPS-TWOHH*CPS)+BB*CPS*CPS
        ax1 = 1./np.sqrt(AAA)
        ax2 = 1./np.sqrt(BBB)
        # area=3.1415927*ax1*ax2

        return ax1, ax2, PS

    @staticmethod
    def get_2d_coords(pos, axes):
        """
        Project along first axis and get coordinates
        on plane defined by remaining two axes.

        Arguments:
            pos (array): position (x,y,z) vector or set of vectors with dimension (n, 3), 
                where n is the number of vectors.
            axes (array): vectors defining 3 axes of coordinate system.

        Returns:
            b_coord (float)
            c_coord (float)
        """

        hA = axes[0]
        hB = axes[1]
        hC = axes[2]
        hA2 = np.repeat(hA, len(pos)).reshape(3, len(pos)).T
        hB2 = np.repeat(hB, len(pos)).reshape(3, len(pos)).T
        hC2 = np.repeat(hC, len(pos)).reshape(3, len(pos)).T

        # find components parallel to hB and hC for coordinates
        b_coord = (pos * hB2 / norm(hB)).sum(axis=1)
        c_coord = (pos * hC2 / norm(hC)).sum(axis=1)

        return b_coord, c_coord

    @staticmethod
    def cut_data(p, p1, s, q, rvir):
        """ Remove particles that fall outside ellipsoid defined
        by new inertia tensor

        Arguments:
            p: (array) particle coordinates in original coordinate system with shape = (3, n)
            p1: (array) particle coordinates in principal axes coordinate system with shape = (3, n)
            s: short axis to long axis ratio
            q: intermediate axis to long axis ratio
            rvir: virial radius or max value for eigenvalues

        Returns:
            new_p: (array) trimmed down particle coordinates in original coordinate system
            new_p1: (array) trimmed down particle coordinates in principal axes coordinate system
        """

        # calculate particle distances in new coord system
        d = p1[0]**2 + (p1[1]/q)**2 + (p1[2]/s)**2
        cut = d < 0.00017**2
        d[cut] = 0.00017**2  # particle distances should not be below force resolution

        # determine which are within the bounds
        cut = d <= (rvir**2)
        # trimmed down in principal axes coordinate system
        new_p1 = p1.T[cut].T
        new_p = p.T[cut].T  # in orig coordinate system

        return new_p, new_p1

    @staticmethod
    def get_inertia_tensor(p, s=1.0, q=1.0, p1=None, normalize=False):
        """ Calculate inertia tensor from particles
        (assuming equal mass particles)

        Arguments:
            p: position vector (x,y,z) for particle or set of particles
              with dimension (3, n), where n is the number of vectors
            s: (default: 1) axis ratio (short/long)
            q: (default: 1) axis ratio (mid/long)
            p1: (default: None) 
                particle coordinates in principal axes coordinate system 
                with dimension (3, n), where n is the number 
                of vectors in frame of principle axis
            normalize: (default: True) whether to normalize by particle distance
                from center

        Returns:
            I: (array) inertia tensor 
        """

        if normalize == True:
            r2 = (p1[0]**2 + (p1[1]/q)**2 + (p1[2]/s)**2)

        else:
            r2 = 1.0

        Ixx = np.sum((p[0]*p[0])/r2)
        Iyy = np.sum((p[1]*p[1])/r2)
        Izz = np.sum((p[2]*p[2])/r2)
        Ixy = np.sum((p[0]*p[1])/r2)
        Iyz = np.sum((p[1]*p[2])/r2)
        Ixz = np.sum((p[0]*p[2])/r2)
        Iyx = Ixy
        Izy = Iyz
        Izx = Ixz
        I = np.array(((Ixx, Ixy, Ixz), (Iyx, Iyy, Iyz), (Izx, Izy, Izz)))

        return I

    @staticmethod
    def fit_inertia_tensor(p, rvir, normalize=True):
        """ Iterative routine to find inertia tensor based on Zemp et. al. 2011

        Arguments:
            p: position vector (x,y,z) for particle or set of particles
              with dimension (3, n), where n is the number of vectors
            rvir: (float) virial radius of host halo
            normalize: (default: True) whether to normalize by particle distance
                from center. Literature recommends normalizing

        Returns:
            I: (array) inertia tensor
        """

        s, q = 1., 1.
        I = HaloOrientation.get_inertia_tensor(
            p, s, q, p1=p, normalize=normalize)
        tol = .001
        it = 0
        err = 1.

        while err > tol:
            s_old, q_old = s, q

            # get eigen vectors and values of inertia tensor
            w, v = HaloOrientation.get_eigs(I, rvir)

            # check if vectors are orthonormal
            HaloOrientation.check_ortho(v)

            # get new s and q
            s, q = HaloOrientation.get_axis_ratios(w)

            # rotate to frame of principle axis
            p1 = HaloOrientation.transform(p, v)

            # select which particles fall within new ellipsoid
            p, p1 = HaloOrientation.cut_data(p, p1, s, q, rvir)

            # recalculate inertia tensor
            I = HaloOrientation.get_inertia_tensor(
                p, s, q, p1, normalize=normalize)

            # compare err to tolerance
            err1 = abs(s_old-s)/s_old
            err2 = abs(q_old-q)/q_old
            err = max(err1, err2)

            it += 1

            if it > 9:
                return I

        return I
