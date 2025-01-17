import numpy as np
from numpy.linalg import norm
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed
import glob
from scipy.optimize import curve_fit
from lenstronomy.Cosmo import nfw_param
import os
import pandas as pd

import argparse

from halo_orientation_2 import HaloOrientation

import itertools
import symlib

# Read in arguments
parser = argparse.ArgumentParser(description="")

parser.add_argument("--ncore", type=int, default=1,
                    help="number of cpus")

parser.add_argument("--data", type=str,
                    help="which symphony data set")

parser.add_argument("--particles", type=str, default="All",
                    help="which grouping of particles to use (All, Host, Subhalos)")

parser.add_argument("--snap", type=int,
                    help="maximum snap number for the simulation (199 or 235)")

parser.add_argument("--nbin", type=int, default=1000,
                    help="pixel resolution of kappa")

parser.add_argument("--rad", type=int, default=1,
                    help="defines maximum radius for kappa calc (rad=4 means rvir/4)")

args = parser.parse_args()

n_core = args.ncore
data_set = args.data
particle_set = args.particles
max_snap_number = args.snap
nbin = args.nbin
rad_cut = args.rad

z_cosmo = 0

if data_set == 'SymphonyLMC':
    host_I = np.load("SymphonyLMC_All_inertia_tensor.npy")
    particle_mass = (5e4)*u.Msun  # Msun
    cosmo = FlatLambdaCDM(H0=70, Om0=0.286)

elif data_set == 'SymphonyMilkyWay':
    host_I = np.load("SymphonyMilkyWay_Host_inertia_tensor_0it_no_norm.npy")
    particle_mass = (4e5)*u.Msun  # Msun
    cosmo = FlatLambdaCDM(H0=70, Om0=0.286)

elif data_set == 'SymphonyGroup':
    host_I = np.load("SymphonyGroup_All_inertia_tensor.npy")
    particle_mass = (3.3e6)*u.Msun  # Msun
    cosmo = FlatLambdaCDM(H0=70, Om0=0.286)

elif data_set == 'SymphonyLCluster':
    host_I = np.load("SymphonyLCluster_All_inertia_tensor.npy")
    particle_mass = (2.2e8)*u.Msun  # Msun
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# This is assuming a particular data structure
path = "/ix/azentner/lom31/lom31/"


def sigma_nfw(r, rs, overdens):
    x = r / rs
    return np.piecewise(
        x,
        [x < 1, x == 1, x > 1],
        [
            lambda x: 2
            * rs
            * overdens
            / (x**2 - 1)
            * (1 - 2 / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))),
            lambda x: 2 * rs * overdens / 3.0,
            lambda x: 2
            * rs
            * overdens
            / (x**2 - 1)
            * (1 - 2 / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x)))),
        ],
    )


def rs_fit(r, rvir, particle_mass, nbins=3000):
    # fit 2d NFW profile for scale radius and overdensity
    # calculate mass from fit

    # rvir in Kpc (halo radius)
    # r in Kpc (particle radial position)

    # return rs and normalization (units h^2 Mo/Kpc^3) (density)
    # fit out to 1/4 virial radius
    bin_edges = np.linspace(5, rvir/4., nbins)
    counts, bins = np.histogram(r, bin_edges)

    bin_area = np.pi * (
        bin_edges[1: len(bin_edges)] ** 2 -
        bin_edges[0: len(bin_edges) - 1] ** 2
    )
    rho = counts * particle_mass / bin_area
    bin_cens = (bin_edges[1:nbins] + bin_edges[0: nbins - 1]) / 2
    sigma = np.sqrt(counts) * particle_mass / bin_area
    sigma[np.where(sigma == 0)] = 1
    p, e = curve_fit(
        sigma_nfw, bin_cens, rho, sigma=sigma, bounds=[
            [0.001, 0.001], [2*rvir/3., np.inf]]
    )
    e0, e1 = np.sqrt(np.diag(e))

    return rho, bin_cens, sigma, p[0], p[1], e0, e1


def calc_halo_props(normalization, rs, z=None, cosmo=None):
    # normalization is in units of h^2*Mo/Mpc^3
    # rs in Mpc
    if cosmo == None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    if z == None:
        z = 0.0
    c = nfw_param.NFWParam(cosmo=cosmo).c_rho0(normalization, z)
    m200 = nfw_param.NFWParam(cosmo=cosmo).M200(rs, normalization, c)

    return c, m200


def read_data(sim_dir, max_snap_number=235, particle_set='All'):
    """
    Load particle and halo data.

    Args:
        sim_dir (string): Name of directory where symphony data sets 
        max_snap_number (int, optional): Which snap shot from simulation to use.
            Defaults to 235. Will depend on which simulation is used.
        particle_set (str, optional): Whether to include host + subhalos ("All"), 
            host only ("Host"), or subhalos only ("Subhalos"). Defaults to 'All'.

    Returns:
        coords (array): Particle coordinates. Shape is (3,n) where n is number of coordinates.
        velocities (array): Particle velocity vector. Shape is (3,n) where n is number of vectors.
        rvir (float): Host halo virial radius
    """

    # Read header data needed to fast particle reads
    part = symlib.Particles(sim_dir)

    # Read particle data for final snapshot
    p = part.read(max_snap_number)

    # get some basic host halo info
    r, hist = symlib.read_rockstar(sim_dir)
    host = r[0, -1]  # First halo, last snapshot.
    okay = r[:, -1]['ok']
    rvir = host['rvir']  # in kpc

    # collect coordinates of particles
    coord_list = []

    if particle_set == 'All':
        for i, h in enumerate(p):
            if okay[i] == True:
                coord_list.append(h['x'][h["ok"]])
        coords = np.concatenate(coord_list).T  # shape 3,n

    elif particle_set == 'subhalos':
        # host is 0th index of p
        # skip 0 to exclude host
        for i, h in enumerate(p[1:]):
            if okay[i] == True:
                coord_list.append(h['x'][h["ok"]])

        coords = np.concatenate(coord_list).T  # shape 3,n

    elif particle_set == 'Host':
        coords = p[0]['x'][p[0]['ok'] == True].T

    return coords, rvir


def get_projection(pos, princip_axes, phi=None, theta=None):

    new_axes, angle, phi, theta = HaloOrientation.get_random_axes_and_angles(
        princip_axes, phi, theta)

    hA = new_axes.T[0]
    hB = new_axes.T[1]
    hC = new_axes.T[2]

    rad_dist = HaloOrientation.get_perp_dist(hA, pos)
    x, y = HaloOrientation.get_2d_coords(pos, new_axes.T)
    ax1, ax2, position_angle = HaloOrientation.get_projected_ellipse(
        C=hC, B=hB, A=hA, th=theta, ph=phi)

    return x, y, angle, phi, theta, position_angle, rad_dist, ax1, ax2


def calc_2d_density(coords, rvir, nbin, rad_cut):

    r = np.sqrt(coords[0]**2 + coords[1]**2)
    r_cut = r < rvir/rad_cut
    x_coords = coords[0][r_cut]
    y_coords = coords[1][r_cut]

    counts, bins_x, bins_y = np.histogram2d(x_coords, y_coords, bins=nbin)
    bin_side_length = bins_x[1]-bins_x[0]
    bin_area = (bin_side_length*u.kpc)**2

    # will need to convert to arcsecs after /(LC.dd*1e3*(4.84e-6))
    # also need to divide by e_crit
    bin_width = bin_side_length
    surface_mass_density = (counts*particle_mass/bin_area).to('Msun/kpc^2')

    return surface_mass_density, bin_width


def run_analysis(j, I, rvir, coords, nbin, rad_cut, phi=None, theta=None):

    r = np.sqrt(coords[0]**2+coords[1]**2+coords[2]**2).T  # returns in kpc
    r_cut = r < rvir
    coords = coords[r_cut]

    evalues, princip_axes = HaloOrientation.get_eigs(I, rvir)

    x, y, angle, phi, theta, position_angle, rad_dist, ax1, ax2 = get_projection(
        coords, princip_axes, phi, theta)

    # long_eig, short_eig = HaloOrientation.get_2d_shape([x, y])

    q = ax2/ax1

    true_rho, bins, err, rs, rho, e_rs, e_rho = rs_fit(
        rad_dist, rvir, particle_mass.value, nbins=3000)

    # normalization (rho) should be in units of h^2*Mo/Mpc^3
    # convert rho from 1/Kpc3 by multiplying by 1e9
    # rs should be also be in units of Mpc
    c, m200 = calc_halo_props(rho*1e9, rs*1e-3, z=z_cosmo, cosmo=cosmo)

    surface_mass_density, bin_width = calc_2d_density(
        [x, y], rvir, nbin, rad_cut)

    results = np.array(
        [j, angle, position_angle, bin_width, q, phi, theta,
         rs, rho, c, m200, e_rs, e_rho, surface_mass_density,
         bins, true_rho], dtype='object')

    return results


# halo_names = [p.rstrip(os.sep) for p in glob.glob('Halo*/')]
halo_names = []

halo_names = []
f = open('/ix/azentner/lom31/lom31/SymphonyMilkyWay/MilkyWay_halo_names.txt')
for l in f:
    halo_names.append(l.split()[0])

# theta_array = np.arccos(np.random.random(250))
# phi_array = 2*np.pi*np.random.random(250)

# phis_and_thetas = np.vstack((phi_array, theta_array)).T
# n_angles = len(phis_and_thetas)

results = []


for j, name in enumerate(halo_names):
    print(name)
    theta_array = np.arccos(np.random.random(100))
    phi_array = 2*np.pi*np.random.random(100)

    phis_and_thetas = np.vstack((phi_array, theta_array)).T
    n_angles = len(phis_and_thetas)

    sim_dir = symlib.get_host_directory(path, data_set, name)
    coords, rvir = read_data(
        sim_dir, max_snap_number=max_snap_number, particle_set=particle_set)

    with Parallel(n_jobs=n_core) as parallel:
        out = parallel(
            delayed(run_analysis)(
                j, host_I[j], rvir,
                coords, nbin, rad_cut,
                phi=phis_and_thetas[i][0],
                theta=phis_and_thetas[i][1],
            ) for i in range(n_angles))
    results.append(out)

    all_data = np.vstack(results)
    """
    np.savez(f'{data_set}_100_rand_ang_rad_{rad_cut}_nbin_{nbin}_projection_props2.npz',
             h_id=all_data.T[0],
             angle=all_data.T[1], position_angle=all_data.T[2],
             bin_width=all_data.T[3], q=all_data.T[4],
             phi=all_data.T[5], theta=all_data.T[6],
             rs=all_data.T[7], rho=all_data.T[8], c=all_data.T[9],
             m200=all_data.T[10], e_rs=all_data.T[11], e_rho=all_data.T[12],
             )

    np.save(
        f'{data_set}_100_rand_ang_rad_{rad_cut}_nbin_{nbin}_projection_counts2.npy', all_data.T[13])
    np.savez(f'{data_set}_100_rand_ang_rad_{rad_cut}_nbin_{nbin}_density2.npz',
             bins=all_data.T[14], density=all_data.T[15])

    """
