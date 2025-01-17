import numpy as np

from astropy.io import ascii
import pandas as pd

from lenstronomy.LensModel.lens_model import LensModel

from astropy.constants import G, c
from astropy import units as u

# from LensMOdelClass import LensMOdelClass
import LensModelDistributions
import LensModelExtraMethods
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from time import time


class CreateLensPop():
    """
    Create population of lenses via ray tracing and monte carlo

    Args:
        halo_catalog:
            Catalog with dark matter halo properties.
        lens_catalog:
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

    def __init__(self, cosmo, LensModelClass, halo_catalog, lens_catalog, source_catalog,
                 halo_type, galaxy_type=None, galaxy_function=None,
                 shear=True):

        # LensModelClass.__init__()

        self.LensModelClass = LensModelClass
        self.cosmo = cosmo
        self.halo_type = halo_type
        self.galaxy_type = galaxy_type
        self.shear = shear
        self.halo_catalog = halo_catalog
        self.lens_catalog = lens_catalog
        self.source_catalog = source_catalog
        self.galaxy_function = galaxy_function

    def create_lens_source_pair(self, lens_cat_object, z_bin_min, z_bin_max):
        """
        Given parameters of a lens and redshift range, randomly select a source to place in redshift bin

        Args:
            lens_cat_object (dataframe): element from lens catalog
            z_bin_min (float): lower redshift limit for selecting source
            z_bin_max (float): upper redshift limit for selecting source

        Returns:
            LC
            source_cat_object
            kwargs_lens
            shear_kwargs
            galaxy_kwargs
            lens_geo_params
        """

        lens_geo_params = {}

        z_lens = lens_cat_object['redshift'].values[0]

        lens_geo_params['z_lens'] = z_lens

        # using galaxy catalog, select redshift within bounds of bin for source
        source_cat_object = LensModelDistributions.select_random_object(
            self.source_catalog,
            z_bin_range=[z_bin_min, z_bin_max])

        z_source = source_cat_object['redshift'].values[0]
        lens_geo_params['z_source'] = z_source

        # set up to calculate physical distances to bin edges
        bin_LC = LensCosmo(
            z_lens=z_bin_min, z_source=z_bin_max, cosmo=self.cosmo)

        # distance in comoving coordinates
        z_bin_min_dist = bin_LC.dd*(1+z_bin_min)
        z_bin_max_dist = bin_LC.ds*(1+z_bin_max)

        lens_geo_params['z_bin_min_dist'] = z_bin_min_dist
        lens_geo_params['z_bin_max_dist'] = z_bin_max_dist

        # volume of spherical shell in which bin is located
        tot_bin_vol = (4/3)*np.pi*(z_bin_max_dist**3 - z_bin_min_dist**3)

        lens_geo_params['tot_bin_vol'] = tot_bin_vol

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

    def get_all_kwargs(self, LC, lens_geo_params, lens_cat_object,
                       halo_cat_object, alphas=None, bin_width=None):
        """
        Gather all kwargs needed as input into lenstronomy and define lens model

        Returns:
            LC (object): LensCosmo defined within lenstronomy formalism
            lens_geo_params (dictionary): redshifts and distances of lensing systems
            lens_cat_object (dataframe): parameters of selected galaxy
            halo_cat_object (dataframe): parameters of selected halo
            alphas (array): Defaults to None. Unscaled deflections for lens
            tensor (array): Defaults to None. 4D tensor for calculating deflections
            bin_width (float): Width between kappa bins in arcsecs Mpc        """

        # empty list for all lens keywords
        kwargs_lens = []
        lens_model_list = []

        # if including shear, create kwargs
        if self.shear == True:

            lens_model_list.append("SHEAR")
            # get gammas for shear
            gamma1, gamma2 = LensModelDistributions.get_shear()
            shear_kwargs = self.LensModelClass.shear_create_kwargs(
                gamma1, gamma2)

            kwargs_lens.append(shear_kwargs)

            lens_model = LensModel(lens_model_list,
                                   z_lens=lens_geo_params['z_lens'],
                                   z_source=lens_geo_params['z_source'])

        # if including galaxy, create kwargs
        if self.galaxy_type != None:
            x_coord, y_coord = LensModelExtraMethods.get_galaxy_coordinates(
                lens_geo_params['lens_dist'],
                max_offset=150.0,
                offset_unit="pc"
            )

            # pick function to create kwargs
            f_name = self.galaxy_function+"_create_kwargs"

            function = getattr(self.LensModelClass, f_name)

            kwargs = function(
                LC=LC,
                properties=lens_cat_object,
                x=x_coord,
                y=y_coord,
            )

            # there might be more than one component in galaxy type
            # go through all of them
            for l in self.galaxy_type:

                lens_model_list.append(l)

            for k in kwargs:
                kwargs_lens.append(k)

            lens_model = LensModel(lens_model_list,
                                   z_lens=lens_geo_params['z_lens'],
                                   z_source=lens_geo_params['z_source'])

        if self.halo_type != None:

            if self.halo_type == 'TABULATED_DEFLECTIONS':
                lens_model_list.append('TABULATED_DEFLECTIONS')

                cc = self.create_custom_deflections(LC=LC,
                                                    alphas=alphas,
                                                    bin_width=bin_width)

                kwargs_lens.append({})

                lens_model = LensModel(lens_model_list,
                                       z_lens=lens_geo_params['z_lens'],
                                       z_source=lens_geo_params['z_source'],
                                       numerical_alpha_class=cc)

            elif self.halo_type == "NFW__ELLIPSE":
                lens_model_list.append(self.halo_type)

                # create halo argument
                halo_kwargs = self.LensModelClass.nfw_ellipse_create_kwargs(
                    LC, properties=halo_cat_object
                )

                kwargs_lens.append(halo_kwargs)

                lens_model = LensModel(lens_model_list,
                                       z_lens=lens_geo_params['z_lens'],
                                       z_source=lens_geo_params['z_source'])

            else:
                lens_model_list.append(self.halo_type)

                # pick function to create kwargs
                f_name = self.halo_type+"_create_kwargs"

                function = getattr(self.LensModelClass, f_name)

                # create halo argument
                halo_kwargs = function(LC, properties=halo_cat_object)

                kwargs_lens.append(halo_kwargs)

                lens_model = LensModel(lens_model_list,
                                       z_lens=lens_geo_params['z_lens'],
                                       z_source=lens_geo_params['z_source'])

        return kwargs_lens, shear_kwargs, lens_model

    def ray_trace(self, LC, rs, kwargs_lens, z_source, z_min_dist, z_max_dist, lens_model):
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
            # compute_window=5, grid_scale=0.03,
            center_x=0, center_y=0
        )

        # if there are caustics, find the maximum extent
        # max caustic coordinate will define area on which to place source
        if len(ra_caustic_list) > 0:

            # lenstronomy prints out caustics disjointed sometimes
            # must concatenate first
            x0 = abs(np.concatenate(ra_caustic_list))  # caustic curves
            y0 = abs(np.concatenate(dec_caustic_list))  # caustic curves

            # find max caustic value
            # multiply by 1.05 to make sure it is a little bigger
            r = max(max(x0), max(y0))*1.05  # arcsecs

            r_rad = r*np.pi/(180*3600)  # convert to radians

            # calculate volume of cyllinder containing source
            # cross section will be pi*r**2 where r is in Mpc
            # found using small angle approx
            r_mpc = LC.ds*(1+z_source)*r_rad

            bin_vol = np.pi*(r_mpc**2) \
                * (z_max_dist-z_min_dist)

            # get source coordinates
            RA, DEC = LensModelDistributions.get_source_position_in_caustic(
                r)  # arcsecs

            # find position of images in lens plane
            theta_ra, theta_dec = solver.findBrightImage(
                RA, DEC, kwargs_lens)
            # precision_limit=10 ** (-5),
            # min_distance=0.005,
            # arrival_time_sort=False, verbose=False)# arcsecs

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

            # if there are no caustics then there will be no lensing
        else:
            RA = np.nan
            DEC = np.nan
            theta_ra = []
            theta_dec = []
            mags = []
            r = 0.0
            bin_vol = 0.0

        return RA, DEC, theta_ra, theta_dec, mags, r, bin_vol

    def monte_carlo(self, z_lens_min, z_lens_max, z_source_min,
                    z_source_max, gal_halo_mass_min, gal_halo_mass_max,
                    mass_function_scale=False, deflection_catalog=None):
        """
        Construct lens by randomly selecting different halo/galaxy/shear properties.
        Select random sources and perform ray tracing.

        Args:
            z_lens_min: Minimum lens redshift
            z_lens_max: Maximum lens redshift
            z_source_min: Minimum source redshift
            z_source_max: Maximum source redshift
        gal_halo_mass_min: Minimum mass of halo hosting galaxy
        gal_halo_mass_max: Maximum mass of halo hosting galaxy
            deflection_catalog (array): Unscaled x,y component deflections for each projection.
                                        Defaults to None.

        Returns:
            df_all (DataFrame): Pandas dataframe with all lensing properties (excluding galaxy) and results
        """

        df_all = pd.DataFrame()

        # select random set of halo properties
        halo_cat_object = LensModelDistributions.select_random_object(
            self.halo_catalog,
            mass_key="m200",
            mass_function_scale=mass_function_scale)  # self.halo_catalog.sample()

        if self.halo_type == 'TABULATED_DEFLECTIONS':
            alphas = deflection_catalog[halo_cat_object['projection_id'].values[0]]
            bin_width = halo_cat_object['bin_width'].values[0]

        else:
            alphas = None
            bin_width = None

        # select reference lens redshift
        lens_cat_object = LensModelDistributions.select_random_object(
            self.lens_catalog,
            z_bin_range=[z_lens_min, z_lens_max],
            position_angle_range=[
                halo_cat_object['position_angle'].values[0]-(np.pi*25.0/180),
                halo_cat_object['position_angle'].values[0]-(np.pi*25.0/180)
            ],
            mass_range=[
                gal_halo_mass_min,
                gal_halo_mass_max],
            mass_key="host_halo_mass"
        )

        z_lens = lens_cat_object['redshift'].values[0]

        if z_source_min < z_lens:
            n_bins, z_bins = LensModelExtraMethods.create_source_redshift_bins(
                z_lens, z_source_max)

        else:
            n_bins, z_bins = LensModelExtraMethods.create_source_redshift_bins(
                z_source_min, z_source_max)

        for i in range(n_bins):

            # get z-range
            z_bin_min = z_bins[i]
            z_bin_max = z_bins[i+1]

            one = time()
            LC, lens_geo_params, source_cat_object = self.create_lens_source_pair(
                lens_cat_object,
                z_bin_min,
                z_bin_max
            )
            # print("time lines source pair: {}".format(time()-one))

            one = time()
            kwargs_lens, shear_kwargs, lens_model = self.get_all_kwargs(
                LC,
                lens_geo_params=lens_geo_params,
                lens_cat_object=lens_cat_object,
                halo_cat_object=halo_cat_object,
                alphas=alphas,
                bin_width=bin_width)

            one = time()
            try:
                RA, DEC, theta_ra, theta_dec, mags, r, bin_vol = self.ray_trace(
                    LC,
                    halo_cat_object['rs'],
                    kwargs_lens,
                    lens_geo_params['z_source'],
                    lens_geo_params['z_bin_min_dist'],
                    lens_geo_params['z_bin_max_dist'],
                    lens_model,
                )

            except:
                RA = np.nan
                DEC = np.nan
                theta_ra = []
                theta_dec = []
                mags = []
                r = 0.0
                bin_vol = 0.0

            df = pd.DataFrame(
                {
                    "n_img": len(theta_ra),
                    "s_id": int(source_cat_object["id"].values[0]),
                    "source_DEC": DEC,
                    "source_RA": RA,
                    "z_lens": lens_geo_params["z_lens"],
                    "z_source": lens_geo_params["z_source"],
                    "caustic_radius": r,
                    "source_dist": lens_geo_params['z_source'],
                    "lens_source_dist": lens_geo_params['lens_source_dist'],
                    "total_bin_volume": lens_geo_params['tot_bin_vol'],
                    "bin_volume": bin_vol,
                    "kwargs_lens": [kwargs_lens],
                    "mag_0": np.nan,
                    "mag_1": np.nan,
                    "mag_2": np.nan,
                    "mag_3": np.nan,
                    "mag_4": np.nan,
                    "mag_5": np.nan,
                    "gamma1": np.nan,
                    "gamma1": np.nan,
                },
                index=[0],
            )

            # If there is a halo, save id and projection data in dataframe
            if self.halo_type != None:
                df["h_id"] = int(halo_cat_object['h_id'].values[0])
                df["projection_id"] = halo_cat_object["projection_id"].values[0]
                df["rot_angle"] = halo_cat_object["angle"].values[0]
                df["position_angle"] = halo_cat_object['position_angle'].values[0]

            # If there is shear update values in dataframe
            if self.shear == True:
                df["gamma1"] = shear_kwargs["gamma1"]
                df["gamma2"] = shear_kwargs["gamma2"]

            # If there is a galaxy, save galaxy id in dataframe too
            if self.galaxy_type != None:
                df["g_id"] = int(lens_cat_object["g_id"].values[0])

            # if there were images, update their magnifications in dataframe
            if len(mags) != 0:
                for i in range(len(mags)):
                    df["mag_{}".format(i)] = mags[i]

            df_all = pd.concat((df_all, df))

        return df_all
