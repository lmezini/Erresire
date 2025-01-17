import numpy as np
from scipy.stats import norm
import sys
import random
from colossus.lss import mass_function
from colossus.cosmology import cosmology


def get_source_position_in_caustic(caustic_r):
    """
    Get 2D angular coordinates of source in the plane of lens

    Args:
        caustic_r (float): maximum extent of the caustics (angular unit)
        n_coords (int, optional): number of coordinate sets to return. Defaults to 1.

    Returns:s
        float: RA coordinate
        float: DEC coordinate
    """

    phi_val = random.random() * 2 * np.pi  # azimuthal position
    r_val = random.random() * caustic_r  # radial position

    RA = r_val * np.cos(phi_val)
    DEC = r_val * np.sin(phi_val)

    return RA, DEC


def get_shear():
    """
    return gamma1 and gamma2 arguments for lenstronomy shear

    Returns:
        float: gamma1
        float: gamma2
    """

    x = np.linspace(-0.16, 0.16, 1000)
    shear_pdf = norm.pdf(x, 0.0, 0.05)

    gamma1s = np.random.choice(x, p=shear_pdf / np.sum(shear_pdf))
    gamma2s = np.random.choice(x, p=shear_pdf / np.sum(shear_pdf))

    return gamma1s, gamma2s


def select_random_object(catalog,
                         z_bin_range=[],
                         position_angle_range=[],
                         mass_range=[],
                         mass_key='mass',
                         redshift_key='redshift',
                         position_angle_key='position_angles',
                         mass_function_scale=False):
    """_summary_

    Args:
        galaxy_catalog (pandas dataframe): catalog of galaxies + properties 
        z_bin_range (tuple): minimum and maximum redshifts (ex. [0.5,2.0])
        mass_range (tuple):  minimum and maximum masses
        mass_function_scale (boolian): 
                Whether to weight by halo mass function. Useful for large distribution of mass.
                Only use for sampling based on halo masses or directly from halo catalog.
                If catalog is already weighted by halo mass function then this would
                be redundant.

    Returns:
        dataframe: properties of galaxy
    """

    if len(z_bin_range) != 0:

        z_mask = np.logical_and(catalog[redshift_key].values >= z_bin_range[0],
                                catalog[redshift_key].values < z_bin_range[1])

        catalog = catalog[z_mask]

        if len(catalog) == 0:

            sys.exit("no objects in redshift bin")

    else:
        pass

    if len(position_angle_range) != 0:

        ang_mask = np.logical_and(
            catalog[position_angle_key].values >= position_angle_range[0],
            catalog[position_angle_key].values < position_angle_range[1]
        )
        catalog = catalog[ang_mask]

        if len(catalog) == 0:
            sys.exit("no objects in position angle bin")

    else:
        pass

    if len(mass_range) != 0:
        mass_mask = np.logical_and(catalog[mass_key].values >= mass_range[0],
                                   catalog[mass_key].values < mass_range[1])

        catalog = catalog[mass_mask]

        if len(catalog) == 0:

            sys.exit("no objects in mass bin")

    else:
        pass

    if mass_function_scale == True:
        # evaluate mass function over range of halo masses in catalog
        mf = mass_function.massFunction(
            catalog[mass_key], 0.0, mdef='200m',
            model='tinker08', q_out='dndlnM')
        # randomly select mass from catalog weighted by halo mass function
        rand_mass = np.random.choice(catalog[mass_key], p=mf / np.sum(mf))
        # locate oject in catalog with that mass
        mass_mask = catalog[mass_key] == rand_mass
        catalog_object = catalog[mass_mask]
        # at this point there will be one remaining object in catalog
        # doing catalog.sample() will return that one object

    catalog_object = catalog.sample()

    return catalog_object


def get_position_angle(max_angle=25, dist_type='uniform'):
    """Select a position angle, phi, for lens components

    Args:
        max_angle (float): in degrees, 0 to 180. Defaults to 25 degrees.
        dist_type (str, optional): Type of distribution to sample angles from.
        Defaults to 'uniform'.

    Returns:
        _type_: _description_
    """

    # convert to radians
    max_angle = max_angle*np.pi/180

    if dist_type == 'uniform':
        return max_angle*random.random()