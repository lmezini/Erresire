### Lens Catalog Attribute Table

#### Lens System Configuration

| Attribute name     | Type    | Definition |
|--------------------|---------|------------|
| `id`               | `int`   | Index of the lens system modeled |
| `s_id`             | `int`   | Index of the source from the source catalog |
| `g_id`             | `int`   | Index of the galaxy from the galaxy catalog |
| `h_id`             | `int`   | Index of the halo from the completed halo mass catalog |
| `projection_id`    | `int`   | Index identifying the halo projection used |
| `flag`             | `int`   | Lens flag (`0` = no images, `1` = images detected) |
| `z_source`         | `float` | Redshift of the source |
| `z_lens`           | `float` | Redshift of the lens |
| `source_DEC`       | `float` | Source DEC coordinate in the lens plane |
| `source_RA`        | `float` | Source RA coordinate in the lens plane |
| `source_dist`      | `float` | Comoving distance to the source (Mpc) |
| `lens_source_dist` | `float` | Comoving distance between the lens and source (Mpc) |
| `Mag_i`            | `float` | Magnitude of the *i-th* lensed image |
| `theta_ra_i`       | `float` | RA coordinate of the *i-th* image in the image plane |
| `theta_dec_i`      | `float` | DEC coordinate of the *i-th* image in the image plane |
| `rot_angle`        | `float` | Angle between halo projection axis and halo major axis |
| `gamma1`           | `float` | First external shear component |
| `gamma2`           | `float` | Second external shear component |
| `n_img`            | `int`   | Number of lensed images detected |



#### Double Sérsic Profile Parameters

| Attribute        | Type    | Definition |
|------------------|---------|------------|
| `k_eff1`         | `float` | Convergence at effective radius of the first Sérsic profile |
| `R_sersic1`      | `float` | Effective radius of the first Sérsic component (arcsec) |
| `n_sersic1`      | `int`   | Sérsic index of the first component |
| `center_x1`      | `float` | x-coordinate of the first component in the lens plane |
| `center_y1`      | `float` | y-coordinate of the first component in the lens plane |
| `e11`            | `float` | First ellipticity component of the first Sérsic profile |
| `e21`            | `float` | Second ellipticity component of the first Sérsic profile |
| `k_eff2`         | `float` | Convergence at effective radius of the second Sérsic profile |
| `R_sersic2`      | `float` | Effective radius of the second Sérsic component (arcsec) |
| `n_sersic2`      | `int`   | Sérsic index of the second component |
| `center_x2`      | `float` | x-coordinate of the second component in the lens plane |
| `center_y2`      | `float` | y-coordinate of the second component in the lens plane |
| `e12`            | `float` | First ellipticity component of the second Sérsic profile |
| `e22`            | `float` | Second ellipticity component of the second Sérsic profile |

#### NFW Ellipse Parameters

| Parameter     | Type    | Definition |
|---------------|---------|------------|
| `Rs`          | `float` | Scale radius of the NFW profile (arcsec) |
| `alpha_Rs`    | `float` | Deflection magnitude at the scale radius |
| `e13`         | `float` | First ellipticity component of the NFW halo |
| `e23`         | `float` | Second ellipticity component of the NFW hal


### Input Catalogs Attribute Table

#### Halo Catalog

| Attribute name   | Type    | Definition |
|------------------|---------|------------|
| `h_id`           | `int`   | ID of object in halo catalog |
| `cat_id`         | `int`   | Index of the halo in the symphony simulation data |
| `rs`             | `float` | Scale radius of the halo (Mpc) |
| `rho`            | `float` | 2D NFW normalization |
| `c`              | `float` | Concentration measured from 2D projected profile |
| `c_intrinsic`    | `float` | Intrinsic concentration of halo in 3D |
| `m200`           | `float` | Halo mass (Solar Masses) |
| `q`              | `float` | Ratio of projected short to long axis |
| `angle`          | `float` | Angle between axis of projection and major axis of halo (radians) |
| `position angle` | `float` | Angle between projected long axis and x-axis (measured counterclockwise) |
| `phi`            | `float` | Azimuthal angle between major axis and projection axis |
| `theta`          | `float` | Polar angle between major axis and projection axis |
| `bin_with`       | `float` | **Used for convergence maps in numeric model only, width of grid bin|
| `sim_name`       | `str`   | Name of symphony simulation halo is from |


#### Galaxy parameters required by built in mass models
If using multiple mass model components (such as for disk and bulge separately, each component needs these values)

 - Stellar mass
 - Host halo mass
 - Redshift
 - Sersic index
 - Axis ratio
 - Half light radius
 