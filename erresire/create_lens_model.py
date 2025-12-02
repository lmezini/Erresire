import numpy as np
from astropy.io import ascii
import lenstronomy.Util.param_util as param_util
from CustomLensModel import create_custom_angular_deflection
from time import time
from astropy import units as u
from astropy.constants import G, c


class CreateLensModel:
    def __init__(self):
        """
        Create kwargs used as input for lenstronomy for each lens component
        """

    @staticmethod
    def nfw_ellipse_create_kwargs(LC, properties, x, y):
        """
        Create all necessary arguments needed as input for lenstronomy halos
        Need Rs and alpha Rs in arcsecs!

        Args:
            LC (object): lens cosmo (from lenstronomy)
            properties (pandas dataframe): dataframe with lens halo properties

        Returns:
            list: list containing dictionary of args to be used as input for lenstronomy
        """

        # get keys
        keys = properties.keys()

        if np.logical_and('Rs_angle' in keys, 'alpha_Rs' in keys):
            Rs_angle = properties['Rs_angle'].values[0]
            alpha_Rs = properties['alpha_Rs'].values[0]

        else:
            # calculate scale radius and deflection angle at scale radius in arcsecs
            c = properties['c'].values[0]
            m200 = properties['halo_mass'].values[0]

            Rs_angle, alpha_Rs = LC.nfw_physical2angle(M=m200, c=c)

        # ellipticity parameters used by lenstronomy for elliptical NFW
        if np.logical_and('e1' in keys, 'e2' in keys):
            e1 = properties['e1'].values[0]
            e2 = properties['e2'].values[0]

        # if e1 and e2 are not known, calculate using position angle and axis ratio
        else:
            # axis ratio
            q = properties['q'].values[0]
            # major axis direction from E to N, set 0 for just rotating about one axis (z-axis)
            # phi is in radians
            phi = properties['position_angle'].values[0]
            e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)

        # create kwargs
        halo_kwargs = {
            "Rs": Rs_angle,
            "alpha_Rs": alpha_Rs,
            "e1": e1,
            "e2": e2,
            "center_x": x,
            "center_y": y
        }

        return halo_kwargs

    @staticmethod
    def sersic_ellipse_sphere_create_kwargs(
        properties,
        LC,
        x=0.0,
        y=0.0,
    ):
        """
        Create kwarg dictionary for lenstronomy lensmodel
        sersic mass profile

        Args:
            properties (pandas dataframe):
            LC (object): lens cosmo from lenstronomy
            x,y (float): galaxy coordinates

        Returns:
            dict: kwargs for lenstronomy lensmodel
        """

        # R_sersic_ang = LC.phys2arcsec_lens(
        #    properties['sphere_half_light_radius'].values[0])  # Mpc to arcsec at z_lens
        R_sersic_ang = properties['sphere_half_light_radius_arcsec'].values[0]

        k_eff = LC.sersic_m_star2k_eff(
            m_star=properties['sphere_mass_stellar'].values[0], R_sersic=R_sersic_ang, n_sersic=properties['sphere_sersic_index'].values[0]
        )

        e1, e2 = param_util.phi_q2_ellipticity(
            phi=properties['position_angle'].values[0],
            q=properties['sphere_axis_ratio'].values[0])

        kwargs = {
            "k_eff": k_eff,
            "R_sersic": R_sersic_ang,
            "n_sersic": properties['sphere_sersic_index'].values[0],
            "center_x": x,
            "center_y": y,
            "e1": e1,
            "e2": e2,
        }

        return kwargs

    @staticmethod
    def sersic_ellipse_disk_create_kwargs(
        properties,
        LC,
        x=0.0,
        y=0.0,
    ):
        """
        Create kwarg dictionary for lenstronomy lensmodel
        sersic mass profile

        Args:
            properties (pandas dataframe):
            LC (object): lens cosmo from lenstronomy
            x,y (float): galaxy coordinates

        Returns:
            dict: kwargs for lenstronomy lensmodel
        """

        # R_sersic_ang = LC.phys2arcsec_lens(
        #    properties['disk_half_light_radius'].values[0])  # Mpc to arcsec at z_lens
        R_sersic_ang = properties['disk_half_light_radius_arcsec'].values[0]

        k_eff = LC.sersic_m_star2k_eff(
            m_star=properties['disk_mass_stellar'].values[0],
            R_sersic=R_sersic_ang, n_sersic=properties['disk_sersic_index'].values[0]
        )

        e1, e2 = param_util.phi_q2_ellipticity(
            phi=properties['position_angle'].values[0],
            q=properties['disk_axis_ratio'].values[0])

        kwargs = {
            "k_eff": k_eff,
            "R_sersic": R_sersic_ang,
            "n_sersic": properties['disk_sersic_index'].values[0],
            "center_x": x,
            "center_y": y,
            "e1": e1,
            "e2": e2,
        }

        return kwargs

    @staticmethod
    def create_custom_deflections(LC, alphas, bin_width, ks=3, s=0.2):
        """
        Return custom deflection class using pre-tabulated deflections for halo.
        Need to normalize by critical surface density and convert bin width to arcsecs.

        Args:
            LC (object): lenstronomy LensCosmo
            tensor (array): 4D tensor for calculating deflections
            alphas (array): Dimensionless 2d projected density of lens
            bin_width (float): Width between kappa bins in Mpc
            ks (int, optional): _description_. Defaults to 3.
            s (int, optional): _description_. Defaults to 0.2 . 

        Returns:
            cc (class instance): custom angular deflections class
        """

        e_crit = (LC.ds/LC.dds) * \
            (((c**2 / (4*np.pi*G))) / (LC.dd*u.Mpc)).to('Msun/kpc^2')
        # convert bin_width to arcsec
        bin_width = LC.phys2arcsec_lens(bin_width*1e-3)

        alpha_x = alphas[0]*bin_width/(LC.dd*1e3*(4.84e-6)*e_crit.value)
        alpha_y = alphas[1]*bin_width/(LC.dd*1e3*(4.84e-6)*e_crit.value)

        Nbin = len(alpha_x)
        cc = create_custom_angular_deflection(
            alpha_x, alpha_y, Nbin, bin_width, ks, s)

        return cc
