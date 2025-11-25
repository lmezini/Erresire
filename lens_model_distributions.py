import numpy as np
from scipy.stats import norm
import sys
import random
from colossus.lss import mass_function
from colossus.cosmology import cosmology
import warnings
from scipy.interpolate import interp1d
cosmology.setCosmology('planck18')


def get_source_position_in_caustic(caustic_r, center_x, center_y):
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

    RA = center_x + r_val * np.cos(phi_val)
    DEC = center_y + r_val * np.sin(phi_val)

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


def compute_hmf_weights(catalog, mass_key='mass', redshift_key='redshift',
                        z_grid=np.linspace(0, 3, 50), num_mass_bins=250):
    """
    Compute halo mass function (HMF) weights for a galaxy/halo catalog.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Catalog with mass and redshift columns.
    mass_key : str
        Name of the mass column.
    redshift_key : str
        Name of the redshift column.
    z_grid : array-like
        Redshift grid for HMF computation.
    num_mass_bins : int
        Number of mass bins for HMF.

    Returns
    -------
    weights : np.ndarray
        HMF-based weight for each galaxy in the catalog.
    """
    df = catalog.copy()

    # Mass bins
    # We will compute the HMF at discrete mass bins across the catalog range
    mass_edges = np.logspace(np.log10(df[mass_key].min()),
                             np.log10(df[mass_key].max()),
                             num_mass_bins + 1)
    mass_centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])

    # Compute HMF on redshift grid
    # hmf_grid has shape (len(z_grid), len(mass_centers))
    # Each row = HMF at a specific redshift, each column = mass bin
    hmf_grid = np.array([
        mass_function.massFunction(
            mass_centers, z, mdef='200m', model='tinker08', q_out='dndlnM'
        ) for z in z_grid
    ])

    # Interpolate HMF to galaxy redshifts
    # interp1d creates a continuous function along the redshift axis
    # Inputs:
    #   z_grid: redshifts where HMF is precomputed
    #   hmf_grid: HMF values at each redshift and mass bin
    # axis=0 means interpolation is done along the rows (redshift axis)
    # fill_value='extrapolate' allows evaluation outside the z_grid range
    hmf_interp = interp1d(z_grid, hmf_grid, axis=0, kind='linear',
                          fill_value='extrapolate')

    # Evaluate the interpolated HMF at each galaxy's actual redshift
    # Resulting shape: (num_galaxies, num_mass_bins)
    # Now each galaxy has a HMF array corresponding to all mass bins
    galaxy_hmf = hmf_interp(df[redshift_key].values)

    # Find nearest mass bin for each galaxy
    mass_idx = np.searchsorted(mass_centers, df[mass_key].values) - 1
    mass_idx = np.clip(mass_idx, 0, len(mass_centers) - 1)

    # Assign weight: pick the HMF value corresponding to galaxy's mass bin
    weights = galaxy_hmf[np.arange(len(df)), mass_idx]

    # Normalize weights to sum to 1 (so they can be used as probabilities)
    weights /= weights.sum()

    return weights


def select_random_object(catalog,
                         z_bin_range=[],
                         mass_range=[],
                         mass_key='mass',
                         redshift_key='redshift',
                         mass_function_weights=None):
    """
    Select a random object from a catalog with optional redshift and mass filtering.

    This function samples a single object from a catalog of galaxies or halos. 
    Optional redshift and mass ranges can be applied to restrict the selection. 
    Additionally, for halo catalogs, objects can be weighted according to the halo 
    mass function when ``mass_function_scale=True`` to account for the expected 
    number density of halos of different masses.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Catalog of galaxies or halos. Must include columns specified by ``mass_key`` 
        and ``redshift_key``.
    z_bin_range : list of two floats, optional
        Redshift range [z_min, z_max] for selection. If None, no redshift filtering is applied.
    mass_range : list of two floats, optional
        Mass range [mass_min, mass_max] for selection. If None, no mass filtering is applied.
    mass_key : str, default 'mass'
        Column name representing object mass.
    redshift_key : str, default 'redshift'
        Column name representing object redshift.
    mass_function_weights : np.ndarray, optional
        Array of precomputed weights (same length as catalog). If None, uniform sampling.

    Returns
    -------
    catalog_object : pandas.DataFrame
        A single-row DataFrame containing the properties of the selected object. 
        Returns an empty DataFrame if no objects match the filtering criteria.

    """

    # Apply redshift filtering
    if z_bin_range:
        catalog = catalog[
            (catalog[redshift_key] >= z_bin_range[0]) &
            (catalog[redshift_key] < z_bin_range[1])
        ]
    # Raise error if no objects remain
    if catalog.empty:
        raise ValueError(
            "No objects found in catalog after applying redshift/mass filters."
        )

    # Apply mass filtering
    if mass_range:
        catalog = catalog[
            (catalog[mass_key] >= mass_range[0]) &
            (catalog[mass_key] < mass_range[1])
        ]

    # Raise error if no objects remain
    if catalog.empty:
        raise ValueError(
            "No objects found in catalog after applying redshift/mass filters."
        )

    # Mass-function-weighted selection
    # Sample using weights if provided
    if mass_function_weights is not None:
        # Filter weights to match filtered catalog
        filtered_weights = mass_function_weights[catalog.index]
        filtered_weights /= filtered_weights.sum()  # normalize
        selected_idx = np.random.choice(catalog.index, p=filtered_weights)

        return catalog.loc[[selected_idx]]

    # Randomly sample one object
    return catalog.sample(n=1)
