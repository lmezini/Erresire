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
import sys


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
            z_bin_range=[z_bin_min, z_bin_max],
            mass_function_scale=False
        )

        z_source = source_cat_object['redshift'].values[0]
        lens_geo_params['z_source'] = z_source

        # lens cosmo for lens and source
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

        lens_geo_params['area_at_source'] = 4*np.pi*source_dist**2

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
            bin_width (float): Width between kappa bins in arcsecs Mpc

        """

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

            elif self.halo_type == "NFW_ELLIPSE":
                lens_model_list.append(self.halo_type)

                # create halo argument
                halo_kwargs = self.LensModelClass.NFW_ELLIPSE_create_kwargs(
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

        return kwargs_lens, lens_model

    def ray_trace(self, LC, kwargs_lens, z_source, lens_model):
        """Perform ray tracing to find image positions

        Returns:
            RA (float): Source RA coordinate (arcsecs)
            DEC (float): Source DEC coordinate (arcsecs)
            theta_ra (list): List of image RA coordinates (arcsecs)
            theta_dec (list): List of image DEC coordinates (arcsecs)
            mags (list): List of image magnifications
            caustic_area (float): Area of caustic region (Mpc^2)
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
            caustic_area = np.pi*(r_mpc**2)

            # get source coordinates
            RA, DEC = LensModelDistributions.get_source_position_in_caustic(
                r)  # arcsecs

            # find position of images in lens plane
            theta_ra, theta_dec = solver.findBrightImage(
                RA, DEC, kwargs_lens,
                # precision_limit=10 ** (-5),
                # min_distance=0.005,
                arrival_time_sort=True, verbose=False)  # arcsecs

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
            caustic_area = 0.0

        return RA, DEC, theta_ra, theta_dec, mags, caustic_area

    def create_galaxy_halo_pair(self, z_lens_min, z_lens_max, deflection_catalog=None):
        """
        For a given dark matter halo, select galaxy from
        galaxy catalog with similar mass.
        ** Deflection catalog only needed if using tabulated deflections **


        Args:
            z_lens_min (float): Minimum lens redshift
            z_lens_max (float): Maximum lens redshift
            deflection_catalog (array): Needed if using tabulated deflections
                (Defaults to None)

        Returns:
            halo_cat_object: element from halo catalog
            lens_cat_object: element from source catalog
            alphas (array): deflections from deflection catalog for selected halo
            bin_width (float): width of bin in deflection catalog for selected halo
        """

        # select set of halo properties
        # mass_function_scale True means selection is weighted by halo mass function
        halo_cat_object = LensModelDistributions.select_random_object(
            self.halo_catalog,
            mass_key="m200",
            mass_range=[10**11.95, 10**14.9],
            mass_function_scale=True)

        # if doing tabulated deflections, select coressponding deflections for halo
        if self.halo_type == 'TABULATED_DEFLECTIONS':
            alphas = deflection_catalog[halo_cat_object['projection_id'].values[0]]
            bin_width = halo_cat_object['bin_width'].values[0]

        else:
            alphas = None
            bin_width = None

        # identify mass of selected halo
        halo_mass = np.log10(halo_cat_object['m200'].values[0])

        # select galaxy with similar halo mass
        # Do not weight by halo mass function this time
        lens_cat_object = LensModelDistributions.select_random_object(
            self.lens_catalog,
            z_bin_range=[z_lens_min, z_lens_max],
            mass_range=[
                10**(halo_mass-0.1),
                10**(halo_mass+0.1)],
            mass_key="host_halo_mass",
            mass_function_scale=False
        )

        return halo_cat_object, lens_cat_object, alphas, bin_width

    def monte_carlo(self, z_lens_min, z_lens_max, z_source_min,
                    z_source_max, deflection_catalog=None):
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

        lens_cat_object = []
        count = 0
        while len(lens_cat_object) == 0:

            halo_cat_object, lens_cat_object, alphas, bin_width = self.create_galaxy_halo_pair(
                z_lens_min, z_lens_max, z_source_min,
                z_source_max, deflection_catalog=deflection_catalog
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
        lens_cat_object['position_angle'] = s*np.pi/180 + \
            halo_cat_object['position_angle'].values[0]
        z_lens = lens_cat_object['redshift'].values[0]

        # for the selected lens, randomly select source and ray trace
        # repeat 5 times
        for i in range(5):

            LC, lens_geo_params, source_cat_object = self.create_lens_source_pair(
                lens_cat_object,
                z_lens+0.05,
                10
            )

            kwargs_lens, lens_model = self.get_all_kwargs(
                LC,
                lens_geo_params=lens_geo_params,
                lens_cat_object=lens_cat_object,
                halo_cat_object=halo_cat_object,
                alphas=alphas,
                bin_width=bin_width)

            try:
                RA, DEC, theta_ra, theta_dec, mags, caustic_area = self.ray_trace(
                    LC,
                    halo_cat_object['rs'],
                    kwargs_lens,
                    lens_geo_params['z_source'],
                    lens_model,
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
                    "area_at_source": lens_geo_params['area_at_source'],
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
                df["g_id"] = int(lens_cat_object["g_id"].values[0])

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
        # rename duplicate keys by adding increasing integer value
        s = df_all.columns.to_series().groupby(df_all.columns)
        df_all.columns = np.where(s.transform('size') > 1,
                                  df_all.columns + s.cumcount().add(1).astype(str),
                                  df_all.columns)

        return df_all
