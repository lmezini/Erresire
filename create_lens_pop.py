import numpy as np
import pandas as pd

from lenstronomy.LensModel.lens_model import LensModel

import lens_model_extra_methods
from create_lens_model import CreateLensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from time import time
import sys


class CreateLensPop():
    """
    Create population of lenses via ray tracing and monte carlo

    Args:
        halo_catalog:
            Catalog with dark matter halo properties.
        galaxy_catalog:
            Catalog with galaxy properties and redshifts.
            *** Need even if not including galaxy to get redshifts
        source_catalog:
            Catalog with source properties and redshifts.
        halo_type (str):
            Type of halo profile (based off of lenstronomy options)
            or "TABULATED_DEFLECTIONS" for custom.
        galaxy_type (str):
            Type of galaxy profile.
        galaxy_function (str):
            Profile(s) used to describe galaxy (based off of lenstronomy options)
            example: [["SERSIC_ELLIPSE","SERSIC_ELLIPSE"]]
        shear (bool, optional):
            Whether or not to add shear. Defaults to True.

    """

    def __init__(self, cosmo, halo_catalog, galaxy_catalog, source_catalog,
                 halo_type, galaxy_type, galaxy_function, halo_function,
                 halo_catalog_mass_key='halo_mass',
                 galaxy_catalog_halo_mass_key='halo_mass',
                 galaxy_catalog_redshift_key='redshift',
                 source_catalog_redshift_key='redshift',
                 custom_galaxy_function=None, custom_halo_function=None,
                 shear=True, mass_function_weights=None):

        self.cosmo = cosmo
        self.halo_type = halo_type
        self.galaxy_type = galaxy_type
        self.shear = shear
        self.halo_catalog = halo_catalog
        self.galaxy_catalog = galaxy_catalog
        self.source_catalog = source_catalog
        self.galaxy_function = galaxy_function
        self.halo_function = halo_function
        self.custom_galaxy_function = custom_galaxy_function
        self.custom_halo_function = custom_halo_function
        self.mass_function_weights = mass_function_weights
        self.halo_catalog_mass_key = halo_catalog_mass_key
        self.galaxy_catalog_halo_mass_key = galaxy_catalog_halo_mass_key
        self.galaxy_catalog_redshift_key = galaxy_catalog_redshift_key
        self.source_catalog_redshift_key = source_catalog_redshift_key

        # Check that custom functions are provided if specified
        for i, gfunc in enumerate(self.galaxy_function):
            if gfunc == "CUSTOM_GALAXY" and self.custom_galaxy_function == None:
                raise ValueError(
                    f"Galaxy component '{self.galaxy_type[i]}' is set to CUSTOM_GALAXY "
                    "but no custom function was provided in custom_galaxy_function."
                )

        for i, hfunc in enumerate(self.halo_function):
            if hfunc == "CUSTOM_HALO" and self.custom_halo_function == None:
                raise ValueError(
                    f"Halo component '{self.halo_type[i]}' is set to CUSTOM_HALO "
                    "but no custom function was provided in custom_halo_function."
                )

    def create_galaxy_halo_pair(self, z_lens_min, z_lens_max, deflection_catalog):
        """
        Create galaxy–halo pair to construct lens.

        This function selects a lens galaxy from the lens catalog within a specified
        redshift interval and then selects a corresponding dark matter halo whose
        mass closely matches the halo mass of the chosen galaxy. If the model uses
        precomputed deflection fields (``halo_type == 'TABULATED_DEFLECTIONS'``),
        the function also retrieves the associated deflection angles and radial bin
        width from the supplied deflection catalog.

        Parameters
        ----------
        z_lens_min : float
            Minimum redshift of the lens galaxy selection bin.
        z_lens_max : float
            Maximum redshift of the lens galaxy selection bin.
        deflection_catalog : dict
            A dictionary mapping halo ``projection_id`` values to precomputed
            deflection fields. Used only when ``self.halo_type`` is
            ``'TABULATED_DEFLECTIONS'``.

        Returns
        -------
        halo_cat_object : pandas.DataFrame
            A single-row DataFrame containing the properties of the selected halo.
        galaxy_cat_object : pandas.DataFrame
            A single-row DataFrame containing the properties of the selected
            lens galaxy.
        alphas : ndarray or None
            Precomputed deflection angles for the selected halo if using tabulated
            deflections; otherwise ``None``.
        bin_width : float or None
            Radial bin width associated with the tabulated deflections; ``None``
            if not using tabulated halo models.

        Notes
        -----
        - The halo is selected such that its mass ``halo_mass`` lies within ±0.01 dex
        of the galaxy's halo mass.
        """

        galaxy_cat_object = lens_model_extra_methods.select_random_object(
            self.galaxy_catalog,
            z_bin_range=[z_lens_min, z_lens_max],
            mass_key=self.galaxy_catalog_halo_mass_key,
            redshift_key=self.galaxy_catalog_redshift_key,
            mass_function_weights=self.mass_function_weights
        )

        gal_halo_mass = np.log10(
            galaxy_cat_object[self.galaxy_catalog_halo_mass_key].values[0])

        # select set of halo properties
        halo_cat_object = lens_model_extra_methods.select_random_object(
            self.halo_catalog,
            mass_key=self.halo_catalog_mass_key,
            mass_range=[
                10**(gal_halo_mass-0.01),
                10**(gal_halo_mass+0.01)],)

        if self.halo_type[0] == 'TABULATED_DEFLECTIONS':
            alphas = deflection_catalog[halo_cat_object['projection_id'].values[0]]
            bin_width = halo_cat_object['bin_width'].values[0]

        else:
            alphas = None
            bin_width = None

        return halo_cat_object, galaxy_cat_object, alphas, bin_width

    def create_lens_source_pair(self, galaxy_cat_object, z_min, z_max):
        """
        Given a lens object and a redshift range, randomly select a source galaxy 
        and compute the geometric and cosmological properties of the lens-source system.

        This method selects a source from the source catalog within the specified redshift bin,
        constructs a `LensCosmo` object to compute cosmological distances, and stores 
        relevant lensing geometry parameters.

        Parameters
        ----------
        galaxy_cat_object : pandas.DataFrame
            Single-row DataFrame representing the lens galaxy, typically from the lens catalog.
            Must include a `redshift` column.
        z_min : float
            Lower bound of the source redshift.
        z_max : float
            Upper bound of the source redshift.

        Returns
        -------
        LC : LensCosmo
            Cosmology object initialized for the lens-source pair.
        lens_geo_params : dict
            Dictionary containing geometric and distance properties of the lens-source system:
            - `z_lens` : redshift of the lens
            - `z_source` : redshift of the source
            - `source_dist` : comoving distance to the source (Mpc)
            - `lens_dist` : comoving distance to the lens (Mpc)
            - `lens_source_dist` : comoving distance between lens and source (Mpc)
        source_cat_object : pandas.DataFrame
            Single-row DataFrame of the selected source galaxy.

        Notes
        -----
        - The method uses `select_random_object` from the source catalog to sample the source redshift.
        - `LensCosmo` is used to compute cosmological distances and lensing geometry.
        - `area_at_source` is computed as the surface area of a sphere at the source's comoving distance.
        """

        lens_geo_params = {}

        z_lens = galaxy_cat_object[self.galaxy_catalog_redshift_key].values[0]

        lens_geo_params['z_lens'] = z_lens

        # using galaxy catalog, select redshift within bounds of bin for source
        source_cat_object = lens_model_extra_methods.select_random_object(
            self.source_catalog,
            redshift_key=self.source_catalog_redshift_key,
            z_bin_range=[z_min, z_max],
        )

        z_source = source_cat_object[self.source_catalog_redshift_key].values[0]
        lens_geo_params['z_source'] = z_source

        # actual lens cosmo for lens and source
        LC = LensCosmo(
            z_lens=z_lens, z_source=z_source, cosmo=self.cosmo)

        # calculate physical distances to lens and source (comoving)
        source_dist = LC.ds*(1+z_source)  # in units of MPc
        lens_dist = LC.dd*(1+z_lens)  # in units of MPc
        # from https://arxiv.org/html/2401.04165v1
        lens_source_dist = LC.dds*(1+z_source)  # in units of MPc

        lens_geo_params['source_dist'] = source_dist
        lens_geo_params['lens_dist'] = lens_dist
        lens_geo_params['lens_source_dist'] = lens_source_dist

        return LC, lens_geo_params, source_cat_object

    def get_all_kwargs(self, LC, lens_geo_params, galaxy_cat_object,
                       halo_cat_object, alphas=None, bin_width=None):
        """
        Gather all keyword arguments required for lenstronomy and construct the lens model.

        This method generates the complete set of keyword arguments (`kwargs_lens`) 
        for the lensing components (shear, galaxy, halo) defined in the instance. 
        It also constructs a `LensModel` object with the specified lens components, 
        cosmological distances, and optional numerical deflections.

        Parameters
        ----------
        LC : LensCosmo
            LensCosmo object containing cosmological distances for the lens-source system.
        lens_geo_params : dict
            Dictionary with geometric parameters of the lensing system, including:
            - `z_lens`: lens redshift
            - `z_source`: source redshift
            - `lens_dist`, `source_dist`, `lens_source_dist`, etc.
        galaxy_cat_object : pandas.DataFrame
            Single-row DataFrame of the selected lens galaxy properties.
        halo_cat_object : pandas.DataFrame
            Single-row DataFrame of the selected halo properties.
        alphas : array-like, optional
            Unscaled deflection angles for the halo (used if `halo_type='TABULATED_DEFLECTIONS'`).
            Default is None.
        bin_width : float, optional
            Width between kappa bins in arcseconds or Mpc for tabulated deflections. Default is None.

        Returns
        -------
        kwargs_lens : list of dict
            List of dictionaries containing keyword arguments for each lens component 
            (shear, galaxy, halo) compatible with lenstronomy.
        lens_model : LensModel
            lenstronomy `LensModel` object constructed from the lens components and their 
            associated cosmological distances.

        Notes
        -----
        - If `self.shear` is True, shear kwargs are included in `kwargs_lens`.
        - Galaxy kwargs are created for each component in `self.galaxy_type` using the 
        corresponding `*_create_kwargs` function.
        - Halo kwargs depend on `self.halo_type`. Special handling exists for:
            - 'TABULATED_DEFLECTIONS': creates a `numerical_alpha_class` with custom deflections.
            - 'NFW_ELLIPSE': uses `NFW_ELLIPSE_create_kwargs` to generate parameters.
            - Other types: uses corresponding `*_create_kwargs` function dynamically.
        - `lens_model` is updated at each step to include all current components.
        """

        # empty list for all lens keywords
        kwargs_lens = []
        lens_model_list = []

        # if including shear, create kwargs
        if self.shear == True:

            lens_model_list.append("SHEAR")
            # get gammas for shear
            gamma1, gamma2 = lens_model_extra_methods.get_shear()
            shear_kwargs = {"gamma1": gamma1, "gamma2": gamma2}

            kwargs_lens.append(shear_kwargs)

        # -----------------------------------------------------------
        # 2. GALAXY COMPONENTS
        # -----------------------------------------------------------
        # Random galaxy position
        x_coord, y_coord = lens_model_extra_methods.get_galaxy_coordinates(
            LC.dd)

        for i, gtype in enumerate(self.galaxy_type):
            lens_model_list.append(gtype)

            if self.galaxy_function[i] == "CUSTOM_GALAXY":
                fn = self.custom_galaxy_function

            else:
                gfunc_name = self.galaxy_function[i]
                fn = getattr(
                    CreateLensModel,
                    gfunc_name + "_create_kwargs"
                )

            # Build kwargs
            galaxy_kwargs = fn(
                LC=LC,
                properties=galaxy_cat_object,
                x=x_coord,
                y=y_coord,
            )

            # Each component yields one kwargs dict, append it
            kwargs_lens.append(galaxy_kwargs)

        # -----------------------------------------------------------
        # 3. HALO COMPONENTS
        # -----------------------------------------------------------
        if self.halo_type[0] == "TABULATED_DEFLECTIONS":
            lens_model_list.append(self.halo_type)
            cc = self.create_custom_deflections(
                LC=LC, alphas=alphas, bin_width=bin_width
            )
            kwargs_lens.append({})  # lenstronomy-required placeholder
            halo_model = LensModel(
                lens_model_list,
                z_lens=lens_geo_params["z_lens"],
                z_source=lens_geo_params["z_source"],
                numerical_alpha_class=cc,
            )
            return kwargs_lens, halo_model

        else:
            for i, htype in enumerate(self.halo_type):
                lens_model_list.append(htype)

                if self.halo_function[i] == "CUSTOM_HALO":
                    fn = self.custom_halo_function

                else:
                    hfunc_name = self.halo_function[i]
                    fn = getattr(
                        CreateLensModel,
                        hfunc_name + "_create_kwargs"
                    )

                # Build kwargs
                halo_kwargs = fn(
                    LC=LC,
                    properties=halo_cat_object,
                    x=0.0,
                    y=0.0,
                )

                kwargs_lens.append(halo_kwargs)

        lens_model = LensModel(lens_model_list,
                               z_lens=lens_geo_params['z_lens'],
                               z_source=lens_geo_params['z_source'])

        return kwargs_lens, lens_model

    def ray_trace(self, kwargs_lens, lens_model):
        """Perform ray tracing to find image positions

        Returns:
            RA (float):
            DEC (float): 
            theta_ra:
            theta_dec:
            mags:
            r:
            bin_vol:
        """

        solver = LensEquationSolver(lens_model)
        lensModelExt = LensModelExtensions(lens_model)

        # calculate coordinates of ciritical and caustic curves
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
            kwargs_lens,
            center_x=0, center_y=0
        )

        # if there are caustics, find the maximum extent
        # max caustic coordinate will define area on which to place source
        if len(ra_caustic_list) > 0:
            try:
                caustic_area, centroid = lens_model_extra_methods.get_caustic_area(
                    ra_caustic_list, dec_caustic_list)
                center_x, center_y = centroid
                r = 2.5

                # get source coordinates
                # use largest possible caustic radius based on most massive halos
                RA, DEC = lens_model_extra_methods.get_source_position_in_caustic(
                    r, center_x=center_x, center_y=center_y)  # arcsecs

                # find position of images in lens plane
                theta_ra, theta_dec = solver.findBrightImage(
                    RA, DEC, kwargs_lens,
                    # precision_limit=10 ** (-5),
                    # min_distance=0.005,
                    arrival_time_sort=False, verbose=False)  # arcsecs

                # solve for magnification of images
                mags = LensModelExtensions(lens_model).magnification_finite(
                    theta_ra,
                    theta_dec,
                    kwargs_lens,
                    source_sigma=0.003,
                    window_size=0.1,
                    grid_number=100,
                    polar_grid=False,
                    aspect_ratio=0.5,
                )

            except:
                RA = np.nan
                DEC = np.nan
                theta_ra = []
                theta_dec = []
                mags = []
                r = 0.0
                caustic_area = 0.0

        return RA, DEC, theta_ra, theta_dec, mags, r, caustic_area

    def monte_carlo(self, z_lens_min, z_lens_max, z_source_max,
                    deflection_catalog=None):
        """
        Construct lens by randomly selecting different halo/galaxy/shear properties.
        Select random sources and perform ray tracing.

        Args:
            z_lens_min: Minimum lens redshift
            z_lens_max: Maximum lens redshift
            z_source_min: Minimum source redshift
            z_source_max: Maximum source redshift
            deflection_catalog (array): Unscaled x,y component deflections for each projection.
                                        Defaults to None.

        Returns:
            df_all (DataFrame): Pandas dataframe with all lensing properties (excluding galaxy) and results
        """

        df_all = pd.DataFrame()

        galaxy_cat_object = []
        count = 0
        while len(galaxy_cat_object) == 0:

            halo_cat_object, galaxy_cat_object, alphas, bin_width = self.create_galaxy_halo_pair(
                z_lens_min, z_lens_max,
                deflection_catalog=deflection_catalog,
            )

            # attempt at most 10 times, if still can't find match
            # then likely will need to change galaxy and halo catalogs
            # so that there is more overlap

            if count == 10:
                sys.exit('unable to match galaxy to halo')
            else:
                count += 1

        # select galaxy position angle offset from halo
        mu, sigma = 0, 10  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)

        # galaxy position angle is halo position angle + offset
        galaxy_cat_object['position_angle'] = s*np.pi/180 + \
            halo_cat_object['position_angle'].values[0]
        z_lens = galaxy_cat_object['redshift'].values[0]

        for i in range(5):

            LC, lens_geo_params, source_cat_object = self.create_lens_source_pair(
                galaxy_cat_object,
                z_lens+0.05,
                z_source_max,
            )

            kwargs_lens, lens_model = self.get_all_kwargs(
                LC,
                lens_geo_params=lens_geo_params,
                galaxy_cat_object=galaxy_cat_object,
                halo_cat_object=halo_cat_object,
                alphas=alphas,
                bin_width=bin_width)

            try:
                RA, DEC, theta_ra, theta_dec, mags, r, caustic_area = self.ray_trace(
                    kwargs_lens, lens_model
                )

            except:
                RA = np.nan
                DEC = np.nan
                theta_ra = []
                theta_dec = []
                mags = []
                r = 0.0
                caustic_area = 0.0

            df = pd.DataFrame(
                {
                    "n_img": len(theta_ra),
                    "s_id": int(source_cat_object["id"].values[0]),
                    "source_DEC": DEC,
                    "source_RA": RA,
                    "z_lens": lens_geo_params["z_lens"],
                    "z_source": lens_geo_params["z_source"],
                    "caustic_radius": r,
                    "source_dist": lens_geo_params['source_dist'],
                    "lens_source_dist": lens_geo_params['lens_source_dist'],
                    "caustic_area": caustic_area,
                    "mag_0": np.nan,
                    "mag_1": np.nan,
                    "mag_2": np.nan,
                    "mag_3": np.nan,
                    "mag_4": np.nan,
                    "mag_5": np.nan,
                    "theta_ra_0": np.nan,
                    "theta_ra_1": np.nan,
                    "theta_ra_2": np.nan,
                    "theta_ra_3": np.nan,
                    "theta_ra_4": np.nan,
                    "theta_ra_5": np.nan,
                    "theta_dec_0": np.nan,
                    "theta_dec_1": np.nan,
                    "theta_dec_2": np.nan,
                    "theta_dec_3": np.nan,
                    "theta_dec_4": np.nan,
                    "theta_dec_5": np.nan,
                },
                index=[0],
            )

            # If there is a halo, save id and projection data in dataframe
            if self.halo_type != None:
                df["h_id"] = int(halo_cat_object['h_id'].values[0])
                df["cat_id"] = int(halo_cat_object['cat_id'].values[0])
                df["projection_id"] = halo_cat_object["projection_id"].values[0]
                df["rot_angle"] = halo_cat_object["angle"].values[0]

            # If there is a galaxy, save galaxy id in dataframe too
            if self.galaxy_type != None:
                df["g_id"] = int(galaxy_cat_object["g_id"].values[0])

            # if there were images, update their magnifications in dataframe
            if len(mags) != 0:
                for i in range(len(mags)):
                    df["mag_{}".format(i)] = mags[i]
                    df["theta_ra_{}".format(i)] = theta_ra[i]
                    df["theta_dec_{}".format(i)] = theta_dec[i]

            # include lens model kwargs
            for kwarg in kwargs_lens:
                kwargs_df = pd.DataFrame.from_dict([kwarg])
                df = pd.concat((df, kwargs_df), axis=1)
            df_all = pd.concat((df_all, df))
            # if lens has multiple components, some keys may be duplicated
            # rename duplicate keys
        s = df_all.columns.to_series().groupby(df_all.columns)
        df_all.columns = np.where(s.transform('size') > 1,
                                  df_all.columns + s.cumcount().add(1).astype(str),
                                  df_all.columns)

        return df_all
