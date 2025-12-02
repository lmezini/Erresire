import numpy as np
from create_lens_model import CreateLensModel
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import warnings
from scipy.stats import norm
import random
from colossus.lss import mass_function
from colossus.cosmology import cosmology
import warnings
from scipy.interpolate import interp1d
cosmology.setCosmology('planck18')


def get_caustic_area(ra, dec):
    """
    Compute the largest polygonal caustic area and its centroid.

    Given lists (or arrays) of RA and Dec coordinates representing one or
    more closed caustic curves, this function constructs a polygon for each
    curve and computes its area. Self-intersecting polygons are automatically
    corrected via ``buffer(0)``. The function returns the area of the largest
    valid caustic polygon and the centroid of that polygon.

    Parameters
    ----------
    ra : array-like
        A sequence of arrays, where each inner array contains the right ascension
        coordinates of a caustic curve. Each element ``ra[j]`` should contain the
        full ordered set of RA points for polygon *j*.
    dec : array-like
        A sequence of arrays matching ``ra``, where ``dec[j]`` contains the
        corresponding declination coordinates for polygon *j*.

    Returns
    -------
    area : float
        The area of the largest valid caustic polygon.
    centroid : tuple of float
        The (RA, Dec) coordinates of the centroid of the largest polygon.

    Notes
    -----
    - Each ``ra[j]`` and ``dec[j]`` must have the same shape and represent an
      ordered boundary of a closed polygon.
    - If a polygon is self-intersecting, it is repaired using
      ``Polygon(...).buffer(0)``.
    - The centroid returned corresponds to the polygon with the maximum area.
    - Input coordinates are assumed to be in consistent units (typically arcseconds).

    """

    area = 0.0
    for j in range(len(ra)):
        # points.append(np.array([ra_caustic_list[j], dec_caustic_list[j]]))
        points = np.array(((ra[j]),
                           (dec[j]))).T
        poly1 = Polygon(points)
        # Check validity — sometimes convexify to fix self-intersections
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        # Compute areas
        area1 = poly1.area

        # Pick larger shape
        if area1 >= area:
            area = area1
        centroid = poly1.centroid.coords[0]

    return area, centroid


def get_concentric_cluster_radius(points, eps=0.15, min_samples=5, centroid_tol=0.02):
    """
    Detects the main cluster and checks for another cluster with the same centroid after removing it.
    If a second concentric cluster exists, returns its radius and the centroid; 
    otherwise returns the radius and centroid of the first cluster.

    Args:
        points (ndarray): Nx2 array of (x, y) coordinates.
        eps (float): DBSCAN neighborhood radius.
        min_samples (int): Minimum samples to form a cluster.
        centroid_tol (float): Distance threshold to consider centroids 'the same'.

    Returns:
        centroid (ndarray): The cluster centroid (shared by both clusters if found).
        radius (float): Radius of the circle enclosing the cluster.
        labels (ndarray): Final DBSCAN labels.
    """
    # --- Step 1: Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]
    if not unique_labels:
        raise ValueError("No valid clusters found!")

    # --- Step 2: Compute centroids and clusters
    clusters = []
    centroids = []
    for lbl in unique_labels:
        cluster_pts = points[labels == lbl]
        clusters.append(cluster_pts)
        centroids.append(cluster_pts.mean(axis=0))
    centroids = np.array(centroids)

    # --- Step 3: Take the largest cluster (most points)
    main_idx = np.argmax([len(c) for c in clusters])
    main_cluster = clusters[main_idx]
    main_centroid = centroids[main_idx]
    main_radius = np.max(np.linalg.norm(main_cluster - main_centroid, axis=1))

    # --- Step 4: Remove those points and re-run DBSCAN
    remaining_points = np.array([p for p in points if not np.any(
        np.all(np.isclose(p, main_cluster), axis=1))])
    if len(remaining_points) == 0:
        # No more points to check
        return main_centroid, main_radius, labels

    clustering2 = DBSCAN(
        eps=eps, min_samples=min_samples).fit(remaining_points)
    labels2 = clustering2.labels_
    unique_labels2 = [lbl for lbl in set(labels2) if lbl != -1]

    # --- Step 5: If there are no other clusters, return the first
    if not unique_labels2:
        return main_centroid, main_radius, labels

    # --- Step 6: Check for a second cluster with similar centroid
    for lbl in unique_labels2:
        cluster_pts = remaining_points[labels2 == lbl]
        centroid2 = cluster_pts.mean(axis=0)
        if np.linalg.norm(centroid2 - main_centroid) < centroid_tol:
            # Found a concentric second cluster
            radius2 = np.max(np.linalg.norm(
                cluster_pts - main_centroid, axis=1))
            return main_centroid, radius2, labels2

    # --- Step 7: No concentric cluster found — return first
    return main_centroid, main_radius


def get_galaxy_params_for_kwargs(galaxy_function, galaxy_catalog_object):
    """
    Generate a keyword-argument dictionary for a specific galaxy profile function.

    This function dynamically calls a corresponding keyword-argument constructor
    for a galaxy profile based on the input function name, and extracts required
    parameters from a single-row lens catalog entry. It raises a warning for any 
    required parameters missing in the catalog.

    Parameters
    ----------
    galaxy_function : str
        Name of the galaxy function as defined in the input file (e.g., 'Sersic').
        The function should correspond to a method named
        ``<galaxy_function>_create_kwargs`` within the ``CreateLensModel`` class.
    galaxy_catalog_object : pandas.DataFrame
        A single-row DataFrame representing the galaxy to be modeled. Column names
        should match the parameter names required by the galaxy function.

    Returns
    -------
    func_args : dict
        Dictionary mapping parameter names to values extracted from 
        ``galaxy_catalog_object``. Only parameters present in the catalog are included.

    Notes
    -----
    - The function assumes ``galaxy_catalog_object`` has exactly one row.
    - If the galaxy function requires parameters not present in the catalog, a warning
      is issued for each missing parameter.
    - Parameter names are inferred from the keyword-argument constructor's 
      function signature (``__code__.co_varnames``).
    - This function facilitates dynamic, catalog-driven construction of galaxy profiles
      without hard-coding parameter names.
    """

    # Construct the function name in CreateLensModel
    f_name = galaxy_function + "_create_kwargs"

    # Get the function from the class
    function = getattr(CreateLensModel, f_name)

    # Get argument names from the function
    varnames = function.__code__.co_varnames
    func_args: {}

    for var in varnames:
        if var in galaxy_catalog_object.columns:
            func_args[var] = galaxy_catalog_object[var].values[0]
        else:
            warnings.warn(
                f"Parameter '{var}' required by '{galaxy_function}_create_kwargs' "
                f"is missing in catalog_object.",
                UserWarning
            )

    return func_args


def rename_kwargs_keys(kwargs: dict, key_label: str) -> dict:
    """
    Append a unique label to each key in a keyword-argument dictionary.

    When multiple components of the same type (e.g., lenses or sources) are present,
    it is useful to have distinct keys for database storage or further processing.
    This function adds a user-defined label to the end of each key in the dictionary.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments describing a component (e.g., a galaxy or lens).
    key_label : str
        Label to append to each key to ensure uniqueness.

    Returns
    -------
    new_kwargs : dict
        A new dictionary with the same values as `kwargs`, but with each key
        appended with `_<key_label>`.

    Example
    -------
    {'e1': 0.2, 'e2': 0.01} becomes {'e11': 0.2, 'e21': 0.01}
    """

    return {f"{k}_{key_label}": v for k, v in kwargs.items()}


def get_galaxy_coordinates(lens_distance_ang_diam):
    """
    Draw a random galaxy position in the lens plane near the halo center.

    The galaxy is placed at a random azimuthal angle and at a radial distance 
    drawn from a probability distribution proportional to 
    :math:`r^2 \exp(-r^2 / (2\sigma^2))`. 
    The resulting radius is converted to an angular separation using the 
    angular diameter distance to the lens and expressed in arcseconds.

    This models galaxies concentrated near a halo center with a characteristic 
    radial scale ``sigma = 192 pc`` (a Maxwellian distribution).

    Parameters
    ----------
    lens_distance_ang_diam : float
        Angular diameter distance to the lens (in Mpc). Used to convert 
        physical offsets (pc) into angular coordinates.

    Returns
    -------
    x_coord : float
        Galaxy x-position in arcseconds, relative to the halo center.
    y_coord : float
        Galaxy y-position in arcseconds, relative to the halo center.
    """

    phi = np.random.uniform(0, 2*np.pi)

    # Radii in pc
    radii_pc = np.linspace(0, 1000, 2000)
    sigma = 192.0

    # Maxwellian-like PDF: r^2 exp(-r^2 / (2 sigma^2))
    pdf = radii_pc**2 * np.exp(-radii_pc**2 / (2*sigma**2))
    pdf /= pdf.sum()  # normalize

    # Sample a radius (pc)
    r_pc = np.random.choice(radii_pc, p=pdf)

    # Convert pc to Mpc
    r_mpc = r_pc / 1e6

    # Convert to radians using angular-diameter distance
    theta_rad = r_mpc / lens_distance_ang_diam

    # Convert radians to arcsec
    r_arcsec = theta_rad * 206265.0

    x = r_arcsec * np.cos(phi)
    y = r_arcsec * np.sin(phi)

    return x, y


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
                         redshift_key=None,
                         mass_key=None,
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
