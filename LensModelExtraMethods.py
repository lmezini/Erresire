import numpy as np
from CreateLensModel import CreateLensModel


def get_galaxy_params_for_kwargs(galaxy_function, lens_catalog_object):
    """
    Create arguments for each galaxy profile specified
    in lens model input file

    Args:
        galaxy_function (string): galaxy function defined in input file
        lens_catalog_object (dataframe): galaxy from lens catalog that is to be modeled

    Returns:
        func_args (dict): dictionary with kwargs and values for lenstronomy 
    """

    # pick function to create kwargs
    f_name = galaxy_function+"_create_kwargs"

    function = getattr(CreateLensModel, f_name)

    # get arguments for function
    varnames = function.__code__.co_varnames

    func_args = {}
    for var in varnames:
        # collect values for each parameter from galaxy catalog
        if var in lens_catalog_object.keys():

            func_args.update(
                {var: lens_catalog_object[var].values[0]})
        else:
            pass

    return func_args


def rename_kwargs_keys(kwargs, key_label):
    """
    If more than one of a type of lens or source component, keys for
    properties that are output to database will need to be distinct
    for clarity.

    This method adds a unique user defined "key label" to each key

    Args:
        kwargs (dictionary): current kwargs describing galaxy
        key_label (string): label to be added to current key

    Returns:
        new_kwargs: dictionary using new keywords
    """
    new_keys = []
    keys = kwargs.keys()

    for key in keys:
        new_keys.append(key+"_"+key_label)
        new_kwargs = dict(zip(new_keys, list(kwargs.values())))

    return new_kwargs


def get_galaxy_coordinates(lens_distance, max_offset=150.0, offset_unit="pc"):
    """
    Get coordinates of galaxy in lens plane
    Will be located near halo center

    Args:
        lens_distance (float): distance to lens in comoving units (Mpc)
        max_offset (float, optional): Maximum offset from halo center. Defaults to 150.
        offset_unit (str, optional): Offset can be either a distance or angle.
            If angle, should be in arcsecs, otherwise, should be in pc. If in pc,
            distance will be converted to arcsecs. Defaults to "pc".

    Returns:
        x_coord, y_coord: coordinates in arcsecs
    """

    # need coordinates
    # choose something that might be slightly off-center
    # in literature galaxies are offset from halo center
    # by 100-200 pc, must convert from pc to arcsecs

    if offset_unit == "pc":
        # using small angle approx
        offset_angle = ((max_offset*1e-6)/lens_distance)
        # will be in radians
        # convert to degrees
        offset_angle = offset_angle * (180/np.pi) * 3600

    elif offset_unit == "angle":
        pass

    x_coord = np.random.random() * offset_angle
    y_coord = np.random.random() * offset_angle

    return x_coord, y_coord


def get_params_for_any_lenstronomy_profile(profile, catalog_object):
    """
    Create arguments for any profile specified in model input file
    Input file needs to have necessary parameters using the same naming convention

    Args:
        lenstronomy_class (module): example "lenstronomy.LensModel.Profiles.cnfw.CNFW"
            do not write as string!
        catalog_object (dataframe): item in catalog that is to be modeled
    """

    # get arguments for function
    varnames = getattr(profile, "param_names")
    varnames = [x.lower() for x in varnames]

    func_args = {}
    for var in varnames:
        # collect values for each parameter from galaxy catalog

        if var in catalog_object.keys():
            func_args.update(
                {var: catalog_object[var].values[0]})
        else:
            pass

    return func_args
