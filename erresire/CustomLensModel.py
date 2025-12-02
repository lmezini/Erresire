import numpy as np
from scipy.interpolate import RectBivariateSpline

# defining numerical angular deflection class##################


class CustomLensingClass(object):
    def __init__(self, alpha_map):
        self.alpha_map = alpha_map
        return None

    def __call__(self, x, y, **kwargs):
        alpha_map = self.alpha_map
        return alpha_map(x, y)
#############################################################


def create_custom_angular_deflection(alpha_x, alpha_y, Nbin, bin_width, ks=4, s=0):
    def alpha_func(x, y):

        grid_x = np.linspace(-(Nbin-1)/2, (Nbin-1)/2, Nbin)*bin_width
        grid_y = np.linspace(-(Nbin-1)/2, (Nbin-1)/2, Nbin)*bin_width

        func_x = RectBivariateSpline(
            grid_x, grid_y, alpha_x, kx=ks, ky=ks, s=s)
        func_y = RectBivariateSpline(
            grid_x, grid_y, alpha_y, kx=ks, ky=ks, s=s)

        return np.array([func_x(x, y, grid=False), func_y(x, y, grid=False)])

    ################################################################################
    custom_class = CustomLensingClass(alpha_map=alpha_func)

    return custom_class
