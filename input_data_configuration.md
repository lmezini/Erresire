### Input Data Configuration Requirements and Recommendations

In order for Erresire to smoothly combine information from multiple input catalogs, some simple pre-processing is required.

Erresire reads model component values directly from input dataframes, so each dataframe must contain column names that match those expected by the mass models. 
For the built-in mass model functionality described in the example use case in Mezini et al. 2025, you can refer directly to the documentation in **create_lens_model.py**.
However, If you choose to define custom functions, you have greater flexibility since the dataframe structure only needs to match the requirements of your custom implementation (in addition to the relational keys in the next paragraph).


#### Relational keys within the simulation


Because Erresire assigns halos to galaxies based on mass, the halo-mass keys in the galaxy and halo catalogs must either be specified explicitly or left as the default "halo_mass".
These keys are provided when instantiating the CreateLensPop class (see model_run_example.ipynb in the examples/ directory).

Rather than storing and exporting every parameter from the input catalogs, Erresire only requires unique IDs that allow simulated lens components to be traced back to their source catalogs.
The required identifiers are:

**h_id** for the halo catalog

**g_id** for the galaxy catalog

**s_id** for the source catalog


#### Additional requirements and recommendations

To define lens and source configuration, each catalog must also provide a redshift column. This can be specified when initializing CreateLensPop, or set to the default name "redshift".
