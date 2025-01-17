import numpy as np
# from astropy.io import ascii
import pandas as pd
import json

from CreateLensPop import CreateLensPop
from astropy.cosmology import FlatLambdaCDM
from CreateLensModel import CreateLensModel

from joblib import Parallel, delayed
import sqlite3

import random
import argparse
from datetime import date
import sys

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)

seed = random.randint(0, int(1e9))
np.random.seed(seed)
random.seed(seed)

## READ FIRST##
# on first iteration do not connect to database
# run this code with n_lens = 1 to generate csv file with output
# output is used to create database
# afterwards run code connected to database
# define 'first_run' as either True or False

# Read in arguments for creating lens model
parser = argparse.ArgumentParser(description="")

parser.add_argument("--nlens", type=int, default=1,
                    help="number of lenses to create on one core")

parser.add_argument("--ncore", type=int, default=1,
                    help="number of cores to run on")

parser.add_argument("--model_param_file", type=str,
                    help="dictionary with parameters to be input to lens model")

parser.add_argument("--first_run", type=bool, default=False,
                    help="True if database to write output to does not exist")

parser.add_argument("--init_file_name", type=str, default=None,
                    help="if first run, name of csv file to output dataframe")

args = parser.parse_args()
n_lens = args.nlens
n_core = args.ncore

f = open(args.model_param_file)
jf = json.load(f)

log_file_name = jf['log_file_name']

out_file = open(log_file_name, 'a')
out_file.write("model version: {}\n".format(sys.argv[0]))
first_run = args.first_run


if first_run == True:
    n_lens = 1
    n_core = 1
    initial_file_name = args.init_file_name
    out_file.write('First run: initialized date: {} \n'.format(date.today()))
    out_file.write('input model param file: {} \n'.format(
        args.model_param_file))
    out_file.write('random seed: {} \n'.format(seed))
    out_file.write('number of unique lenses: {} \n'.format(n_lens))
    out_file.write('number of cores used: {} \n'.format(n_core))

else:
    n_lens = args.nlens
    n_core = args.ncore
    database_name = jf['database_name']

    out_file.write('Date: {} \n'.format(date.today()))

    out_file.write('input model param file: {} \n'.format(
        args.model_param_file))
    out_file.write('random seed: {} \n'.format(seed))
    out_file.write('number of unique lenses: {} \n'.format(n_lens))
    out_file.write('number of cores used: {} \n'.format(n_core))

z_min = jf['z_min']
z_max = jf['z_max']

gal_halo_mass_min = 10**jf['gal_halo_mass_min']
gal_halo_mass_max = 10**jf['gal_halo_mass_max']

halo_type = jf['halo_type']
halo_data_file_name = jf['halo_data_file']
# halo_deflections_name = jf['halo_deflection_file']
galaxy_type = jf['galaxy_type']
galaxy_function = jf['galaxy_function']
lens_catalog_name = jf['lens_catalog']
source_catalog_name = jf['source_catalog']
shear = jf['shear']

lens_catalog = pd.read_pickle(lens_catalog_name)
source_catalog = pd.read_pickle(source_catalog_name)
halo_catalog = pd.read_pickle(halo_data_file_name)
# halo_deflections = np.load(halo_deflections_name)['f'][0]

if first_run == True:
    pass

else:
    conn = sqlite3.connect(database_name)
    print('connected')
    c = conn.cursor()


def main(clp):

    df = clp.monte_carlo(z_lens_min=0.0, z_lens_max=1.5,
                         z_source_min=1.75, z_source_max=3.0,
                         gal_halo_mass_min=gal_halo_mass_min,
                         gal_halo_mass_max=gal_halo_mass_max,
                         mass_function_scale=False)

    return df


lens_model_class = CreateLensModel()

clp = CreateLensPop(cosmo=cosmo,
                    LensModelClass=lens_model_class,
                    halo_catalog=halo_catalog,
                    lens_catalog=lens_catalog,
                    source_catalog=source_catalog,
                    halo_type=halo_type,
                    galaxy_type=galaxy_type,
                    galaxy_function=galaxy_function,
                    shear=True)


df_all = pd.DataFrame()

with Parallel(n_jobs=n_core) as parallel:
    results = parallel(
        delayed(main)(clp)
        for n in range(n_lens)
    )

for res in results:
    df_all = pd.concat((df_all, res))

if first_run == True:
    length = 0
else:
    c.execute("""SELECT COUNT(rot_angle) from halo""")
    length = c.fetchall()[0][0]

# every lens gets a unique id
df_all["id"] = length + np.arange(0, len(df_all), 1)

# flag lenses that produced multiple images
df_all["flag"] = df_all["n_img"] >= 1


if first_run == True:
    out_file.write('output initial file: {} \n'.format(initial_file_name))
    df_all.to_csv(initial_file_name)

else:
    # break up data frame into tables
    lens = df_all[
        [
            "id",
            "projection_id",
            "h_id",
            "g_id",
            "s_id",
            "source_DEC",
            "source_RA",
            "z_lens",
            "z_source",
            "source_dist",
            "lens_source_dist",
            "total_bin_volume",
            "bin_volume",
            "caustic_radius",
            "gamma1",
            "gamma2",
            "flag",
        ]
    ]

    all_images = df_all[
        [
            "id",
            "n_img",
            "mag_0",
            "mag_1",
            "mag_2",
            "mag_3",
            "mag_4",
            "mag_5"
        ]
    ]

    image = all_images[all_images["n_img"] >= 0]

    lens.to_sql("lens", conn, if_exists="append", index=False)
    image.to_sql("image", conn, if_exists="append", index=False)

    df_all.to_csv('lens2.csv')
    conn.commit()
