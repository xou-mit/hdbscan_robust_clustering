# Import the libraries and the datasets
print('Loading libraries...')
import numpy as np
import uncertainties
from uncertainties import unumpy
import astropy
from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
coord.galactocentric_frame_defaults.set('v4.0')
import pandas as pd

import warnings
warnings.filterwarnings("once")
#SettingWithCopyWarning

import sys
import timeit
from tqdm import tqdm

print('Loading datasets...')
# Input gaia edr3 file path
input_path = sys.argv[1]

star_hdus = fits.open(input_path)
allstar_gaiae3_arr = star_hdus[1].data
star_hdus.close()

# Dissect the array if needed for testing
# Do the whole DF if both arguments are the same
ind_ini = int(sys.argv[2])
ind_fin = int(sys.argv[3])
if ind_ini < ind_fin:
    allstar_gaiae3_arr = allstar_gaiae3_arr[ind_ini:ind_fin]

star_hdus = fits.open('../data/sdss16_reduced.fits')
allstar_sdss16_arr = star_hdus[1].data
star_hdus.close()

star_hdus = fits.open('../data/lamost6_reduced.fits')
allstar_lamost6lrs_ste_arr = star_hdus[1].data
star_hdus.close()

star_hdus = fits.open('../data/apogee_reduced.fits')
allstar_apogee_arr = star_hdus[1].data
star_hdus.close()

star_hdus = fits.open('../data/galah3_reduced.fits')
allstar_galah3_arr = star_hdus[1].data
star_hdus.close()

allstar_rave6_df = pd.read_csv('../data/ravedr6_param_abun_x_edr3.csv')

# Unique-fy the RAVEDR6 dataframe
allstar_rave6_unique_df = allstar_rave6_df.drop_duplicates(subset = ["source_id"])

print('Converting dataset into DataFrame...')
# Try invert the bytes in the FITS array
# Get the name list
names_list = allstar_gaiae3_arr.names
# Get the dtype list
dtype_list = []
for i in names_list:
    dtype_temp = allstar_gaiae3_arr[i][0].newbyteorder().dtype
    dtype_list.append(dtype_temp)

# Create the new structured array to store the bytes inverted values
allstar_gaiae3_arr_binv_dtype = np.dtype({'names': names_list, 'formats': dtype_list})
allstar_gaiae3_arr_binv = np.empty(allstar_gaiae3_arr.shape, dtype=allstar_gaiae3_arr_binv_dtype)
# print(allstar_gaiae3_arr_binv)

# Store the values into the new array
for i in names_list:
#     print(allstar_gaiae3_arr[i].byteswap().newbyteorder().dtype)
    allstar_gaiae3_arr_binv[i] = allstar_gaiae3_arr[i].byteswap().newbyteorder()
# Try converting edr3 into dataframe
allstar_gaiae3_df = pd.DataFrame(allstar_gaiae3_arr_binv)

# Define the constants needed for velocity conversion
c = 299792.458 # Speed of light [km/sec]
k = 4.74047 # constant of proportionality 
cos = np.cos
sin = np.sin
# J2000 positions of NGP and GC, taken from Jo Bovy's thesis appendix
# Technically these should be converted into ICRS values, as those are what's reported
# for GAIA
# theta at J2000
# theta = 122.932*np.pi/180
ra_NGP,dec_NGP = 192.85948*u.degree, 27.12825*u.degree  
ra_GC, dec_GC  = 266.405*u.degree, -28.936*u.degree

# Convert these to ICRS frame
c_NGP_fk5, c_GC_fk5 = coord.SkyCoord(ra_NGP,dec_NGP,frame="fk5",equinox='J2000'), coord.SkyCoord(ra_GC,dec_GC,frame="fk5",equinox='J2000')
c_NGP_icrs, c_GC_icrs = c_NGP_fk5.transform_to('icrs'), c_GC_fk5.transform_to('icrs')
ra_NGP,dec_NGP = c_NGP_icrs.ra.to(u.rad).value, c_NGP_icrs.dec.to(u.rad).value
ra_GC, dec_GC  = c_GC_icrs.ra.to(u.rad).value, c_GC_icrs.dec.to(u.rad).value

# Calculate the new theta
sin_theta = sin(ra_GC-ra_NGP)*cos(dec_GC)
cos_theta = sin(dec_GC)/cos(dec_NGP)
theta = ((np.pi-np.arcsin(sin_theta))+np.arccos(cos_theta))/2


def celes_to_gc_coord_astropy(ra,dec,plx):
    '''
    Input units as before
    Output coordinates are in pc
    '''
    
    ra= ra*u.deg
    dec= dec*u.deg
    distance= 1/plx*u.kpc
    coor= coord.ICRS(ra=ra,dec=dec,distance=distance)
    
    gc_frame= coord.Galactocentric()

    cg= coor.transform_to(gc_frame)
    cg.representation_type = 'cartesian'
    
    return cg.x.to(u.pc).value, cg.y.to(u.pc).value, cg.z.to(u.pc).value

# Define column names for dataframes
dataset_list = ['GaiaDR2','SDSSDR16','APOGEE','RAVEDR6','LAMOSTDR6LRSSTELLAR','GALAHDR3']
zqul_list = ['zqul_g2','zqul_16','zqul_ap','zqul_r6','zqul_l6s','zqul_gl3']
z_list = ['z_g2','z_16','z_ap','z_r6','z_l6s','z_gl3']
zerr_list = ['zerr_g2','zerr_16','zerr_ap','zerr_r6','zerr_l6s','zerr_gl3']
m_h_list = ['m_h_g2','m_h_16','m_h_ap','m_h_r6','m_h_l6s','m_h_gl3']
m_h_err_list = ['m_h_err_g2','m_h_err_16','m_h_err_ap','m_h_err_r6','m_h_err_l6s','m_h_err_gl3']
m_h_flg_list = ['m_h_flg_g2','m_h_flg_16','m_h_flg_ap','m_h_flg_r6','m_h_flg_l6s','m_h_flg_gl3']
alpha_m_list = ['alpha_m_g2','alpha_m_16','alpha_m_ap','alpha_m_r6','alpha_m_l6s','alpha_m_gl3']
alpha_m_err_list = ['alpha_m_err_g2','alpha_m_err_16','alpha_m_err_ap','alpha_m_err_r6','alpha_m_err_l6s','alpha_m_err_gl3']
alpha_m_flg_list = ['alpha_m_flg_g2','alpha_m_flg_16','alpha_m_flg_ap','alpha_m_flg_r6','alpha_m_flg_l6s','alpha_m_flg_gl3']
U_list = ['U_g2','U_16','U_ap','U_r6','U_l6s','U_gl3']
V_list = ['V_g2','V_16','V_ap','V_r6','V_l6s','V_gl3']
W_list = ['W_g2','W_16','W_ap','W_r6','W_l6s','W_gl3']
Uerr_list = ['Uerr_g2','Uerr_16','Uerr_ap','Uerr_r6','Uerr_l6s','Uerr_gl3']
Verr_list = ['Verr_g2','Verr_16','Verr_ap','Verr_r6','Verr_l6s','Verr_gl3']
Werr_list = ['Werr_g2','Werr_16','Werr_ap','Werr_r6','Werr_l6s','Werr_gl3']
UVWcov_list = ['UVWcov_g2','UVWcov_16','UVWcov_ap','UVWcov_r6','UVWcov_l6s','UVWcov_gl3']
UVcov_list = ['UVcov_g2','UVcov_16','UVcov_ap','UVcov_r6','UVcov_l6s','UVcov_gl3']
UWcov_list = ['UWcov_g2','UWcov_16','UWcov_ap','UWcov_r6','UWcov_l6s','UWcov_gl3']
VWcov_list = ['VWcov_g2','VWcov_16','VWcov_ap','VWcov_r6','VWcov_l6s','VWcov_gl3']
v_forder_flag_list = ['v_forder_g2','v_forder_16','v_forder_ap','v_forder_r6','v_forder_l6s','v_forder_gl3']
best_col = ['best_z','best_zerr','best_zqul','best_U','best_V','best_W','best_Uerr','best_Verr','best_Werr',
            'best_UVcov','best_UWcov','best_VWcov','best_v_forder_flag']
col_arr = np.array([z_list,zerr_list,zqul_list,U_list,V_list,W_list,Uerr_list,Verr_list,Werr_list,
                    UVcov_list,UWcov_list,VWcov_list,v_forder_flag_list])

# Define solar motion
sun_motion = coord.SkyCoord(x=-8.122*u.kpc,y=0*u.pc,z=20.8*u.pc,v_x=12.9*u.km/u.s,v_y=245.6*u.km/u.s,v_z=7.78*u.km/u.s,frame=coord.Galactocentric)
sun_motion_cyl = coord.SkyCoord(x=-8.122*u.kpc,y=0*u.pc,z=20.8*u.pc,v_x=12.9*u.km/u.s,v_y=245.6*u.km/u.s,v_z=7.78*u.km/u.s,frame=coord.Galactocentric)
sun_motion_cyl.representation_type = 'cylindrical'

# Define functions for handling crossmatching and velocities calculation
def cross_match_datasets(input_gaiaedr3_ini,cm_rad = 3*u.arcsec):
    '''
    Function that takes GAIA EDR3 data and
    crossmatch it with the following datasets
    for RV measurements
    with the specified radius.
    Crossmatch with GaiaDR2 and RAVEDR6 are based on already given source_id match
    whereas the rest are based on position.
    The output also includes a summary df that entails
    which datasets provided a match for the stars in the
    list.
    '''
    
    input_gaiaedr3_list = input_gaiaedr3_ini.copy()
    
    N_datasets = len(dataset_list)
    N_stars = len(input_gaiaedr3_list['source_id'])
    
    print("Starting crossmatch...")

    for i in range(N_datasets):
        # Skip the ones that we can match through source_id
        if dataset_list[i] == 'GaiaDR2' or dataset_list[i] == 'RAVEDR6':
            continue
        
        # Put columns of diff datasets z measurements for later comparison
        input_gaiaedr3_list[z_list[i]] = np.nan
        input_gaiaedr3_list[zerr_list[i]] = np.nan
        
        # Put columns of diff datasets m_h and alpha_m measurements for later comparison
        if dataset_list[i] == 'APOGEE' or dataset_list[i] == 'LAMOSTDR6LRSSTELLAR' or dataset_list[i] == 'GALAHDR3':
            input_gaiaedr3_list[m_h_list[i]] = np.nan
            input_gaiaedr3_list[m_h_err_list[i]] = np.nan
            input_gaiaedr3_list[m_h_flg_list[i]] = np.nan
            input_gaiaedr3_list[alpha_m_list[i]] = np.nan
            input_gaiaedr3_list[alpha_m_err_list[i]] = np.nan
            input_gaiaedr3_list[alpha_m_flg_list[i]] = np.nan

    start_time_tmp = timeit.default_timer()
    # For Gaia EDR3 itself, simply use the already included DR2 velocities
    input_gaiaedr3_list["z_g2"] = input_gaiaedr3_list["dr2_radial_velocity"]/c
    input_gaiaedr3_list["zerr_g2"] = input_gaiaedr3_list["dr2_radial_velocity_error"]/c
    input_gaiaedr3_list["zqul_g2"] = abs(input_gaiaedr3_list["zerr_g2"]/input_gaiaedr3_list["z_g2"])
    
    ind_gaia2 = np.where(np.isnan(input_gaiaedr3_list["z_g2"]) == False)[0]
    elapsed_tmp = timeit.default_timer() - start_time_tmp
    
    print('Finished crossmatching GAIA DR2 in ',elapsed_tmp,'sec and found ',len(ind_gaia2),'matches...')

    # Start crossmatching with the specified radius
    c_input = coord.SkyCoord(ra=input_gaiaedr3_list["ra"].values*u.degree, dec=input_gaiaedr3_list["dec"].values*u.degree)
    
    start_time_tmp = timeit.default_timer()
    # Attempt the cross match of input stars with SDSS DR16
    catalog = coord.SkyCoord(ra=allstar_sdss16_arr['flux_ra']*u.degree, dec=allstar_sdss16_arr['flux_dec']*u.degree)
    idx, d2d, d3d = c_input.match_to_catalog_3d(catalog)

    # Add the sdss dr16 redshifts to the data frame
    temp_ind = (d2d < cm_rad)
    ind_sdss16 = np.arange(len(input_gaiaedr3_list))[temp_ind]
    input_gaiaedr3_list.loc[temp_ind,"z_16"] = allstar_sdss16_arr['z'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"zerr_16"] = allstar_sdss16_arr['z_err'][idx[temp_ind]]
    input_gaiaedr3_list["zqul_16"] = abs(input_gaiaedr3_list["zerr_16"]/input_gaiaedr3_list["z_16"])
    
    elapsed_tmp = timeit.default_timer() - start_time_tmp
    print('Finished crossmatching SDSS DR16 in ',elapsed_tmp,'sec and found ',len(ind_sdss16),'matches...')
    
    start_time_tmp = timeit.default_timer()
    # Attempt the cross match of input stars with LAMOST DR6 LRS Stellar
    catalog = coord.SkyCoord(ra=allstar_lamost6lrs_ste_arr['ra']*u.degree, dec=allstar_lamost6lrs_ste_arr['dec']*u.degree)
    idx, d2d, d3d = c_input.match_to_catalog_3d(catalog)

    # Add the LAMOST DR6 Stellar redshifts and metallicity to the data frame
    temp_ind = (d2d < cm_rad)
    ind_lamost6lrs_ste = np.arange(len(input_gaiaedr3_list))[temp_ind]
    input_gaiaedr3_list.loc[temp_ind,"z_l6s"] = allstar_lamost6lrs_ste_arr['z'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"zerr_l6s"] = allstar_lamost6lrs_ste_arr['z_err'][idx[temp_ind]]
    input_gaiaedr3_list["zqul_l6s"] = abs(input_gaiaedr3_list["zerr_l6s"]/input_gaiaedr3_list["z_l6s"])
    input_gaiaedr3_list.loc[temp_ind,"m_h_l6s"] = allstar_lamost6lrs_ste_arr['feh'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"m_h_err_l6s"] = allstar_lamost6lrs_ste_arr['feh_err'][idx[temp_ind]]
    
    elapsed_tmp = timeit.default_timer() - start_time_tmp
    print('Finished crossmatching LAMOST DR6 LRS Stellar in ',elapsed_tmp,'sec and found ',len(ind_lamost6lrs_ste),'matches...')
    
    start_time_tmp = timeit.default_timer()
    # Attempt the cross match of input stars with APOGEE
    catalog = coord.SkyCoord(ra=allstar_apogee_arr['ra']*u.degree, dec=allstar_apogee_arr['dec']*u.degree)
    idx, d2d, d3d = c_input.match_to_catalog_3d(catalog)

    # Add the APOGEE redshifts and metallicity to the data frame
    temp_ind = (d2d < cm_rad)
    ind_apogee = np.arange(len(input_gaiaedr3_list))[temp_ind]
    input_gaiaedr3_list.loc[temp_ind,"z_ap"] = allstar_apogee_arr['vhelio_avg'][idx[temp_ind]]/c
    input_gaiaedr3_list.loc[temp_ind,"zerr_ap"] = allstar_apogee_arr['vscatter'][idx[temp_ind]]/c
    input_gaiaedr3_list["zqul_ap"] = abs(input_gaiaedr3_list["zerr_ap"]/input_gaiaedr3_list["z_ap"])
    input_gaiaedr3_list.loc[temp_ind,"m_h_ap"] = allstar_apogee_arr['m_h'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"m_h_err_ap"] = allstar_apogee_arr['m_h_err'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"m_h_flg_ap"] = allstar_apogee_arr['ASPCAPFLAG'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"alpha_m_ap"] = allstar_apogee_arr['alpha_m'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"alpha_m_err_ap"] = allstar_apogee_arr['alpha_m_err'][idx[temp_ind]]
    
    elapsed_tmp = timeit.default_timer() - start_time_tmp
    print('Finished crossmatching APOGEE in ',elapsed_tmp,'sec and found ',len(ind_apogee),'matches...')
    
    start_time_tmp = timeit.default_timer()
    # Attempt the cross match of input stars with GALAH DR3
    catalog = coord.SkyCoord(ra=allstar_galah3_arr['ra_dr2']*u.degree, dec=allstar_galah3_arr['dec_dr2']*u.degree)
    idx, d2d, d3d = c_input.match_to_catalog_3d(catalog)

    # Add the GALAH redshifts and metallicity to the data frame
    temp_ind = (d2d < cm_rad)
    ind_galah3 = np.arange(len(input_gaiaedr3_list))[temp_ind]
    input_gaiaedr3_list.loc[temp_ind,"z_gl3"] = allstar_galah3_arr['rv_galah'][idx[temp_ind]]/c
    input_gaiaedr3_list.loc[temp_ind,"zerr_gl3"] = allstar_galah3_arr['e_rv_galah'][idx[temp_ind]]/c
    input_gaiaedr3_list["zqul_gl3"] = abs(input_gaiaedr3_list["zerr_gl3"]/input_gaiaedr3_list["z_gl3"])
    input_gaiaedr3_list.loc[temp_ind,"m_h_gl3"] = allstar_galah3_arr['fe_h'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"m_h_err_gl3"] = allstar_galah3_arr['e_fe_h'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"m_h_flg_gl3"] = allstar_galah3_arr['flag_fe_h'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"alpha_m_gl3"] = allstar_galah3_arr['alpha_fe'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"alpha_m_err_gl3"] = allstar_galah3_arr['e_alpha_fe'][idx[temp_ind]]
    input_gaiaedr3_list.loc[temp_ind,"alpha_m_flg_gl3"] = allstar_galah3_arr['flag_alpha_fe'][idx[temp_ind]]
    
    elapsed_tmp = timeit.default_timer() - start_time_tmp
    print('Finished crossmatching GALAH DR3 in ',elapsed_tmp,'sec and found ',len(ind_galah3),'matches...')
    
    start_time_tmp = timeit.default_timer()
    # Attempt the cross match of input stars with RAVE6
    # Using pre-matched result
    # Put the metallicity info in as well
    input_gaiaedr3_list = input_gaiaedr3_list.merge(allstar_rave6_unique_df.loc[:,['source_id','hrv_sparv','hrv_error_sparv',
                                                                                   'fe_h_gauguin','alpha_fe_gauguin',
                                                                                'fe_h_error_gauguin','alpha_fe_error_gauguin']],
                                                    on='source_id',how='left')
    input_gaiaedr3_list["hrv_sparv"] = input_gaiaedr3_list["hrv_sparv"]/c
    input_gaiaedr3_list["hrv_error_sparv"] = input_gaiaedr3_list["hrv_error_sparv"]/c
    input_gaiaedr3_list = input_gaiaedr3_list.rename(columns={'hrv_sparv':'z_r6','hrv_error_sparv':'zerr_r6',
                                                             'fe_h_gauguin':'m_h_r6','alpha_fe_gauguin':'alpha_m_r6',
                                                'fe_h_error_gauguin':'m_h_err_r6','alpha_fe_error_gauguin':'alpha_m_err_r6'})
    input_gaiaedr3_list["zqul_r6"] = abs(input_gaiaedr3_list["zerr_r6"]/input_gaiaedr3_list["z_r6"])
    
    ind_rave6 = np.where(np.isnan(input_gaiaedr3_list["z_r6"]) == False)[0]
    
    elapsed_tmp = timeit.default_timer() - start_time_tmp
    print('Finished crossmatching RAVE DR6 in ',elapsed_tmp,'sec and found ',len(ind_rave6),'matches...')

    # Create a summary data frame for the crossmatch
    cross_match_sum = pd.DataFrame({'GaiaDR2': np.zeros(N_stars), 'SDSSDR16': np.zeros(N_stars),  
                                    'APOGEE': np.zeros(N_stars), 'RAVEDR6': np.zeros(N_stars), 
                                    'LAMOSTDR6LRSSTELLAR': np.zeros(N_stars),
                                   'GALAHDR3': np.zeros(N_stars)})

    # Mark which datasets found a match
    cross_match_sum['GaiaDR2'][ind_gaia2] = 1
    cross_match_sum['SDSSDR16'][ind_sdss16] = 1
    cross_match_sum['APOGEE'][ind_apogee] = 1
    cross_match_sum['RAVEDR6'][ind_rave6] = 1
    cross_match_sum['LAMOSTDR6LRSSTELLAR'][ind_lamost6lrs_ste] = 1
    cross_match_sum['GALAHDR3'][ind_galah3] = 1

    # Count how many matches are found
    cross_match_sum['Sum'] = cross_match_sum.sum(axis=1)

    match_ind = (cross_match_sum["Sum"] != 0)
    N_match = len(cross_match_sum["Sum"][match_ind].values)
    print('Finished crossmatching with',N_match,'stars gaining matches out of',len(input_gaiaedr3_list))
    
    # Drop the rows of the gaia dataset and the summary dataframes with no radial velocity measurements
    nonmatch_ind = np.where(cross_match_sum["Sum"].values == 0)[0]
    input_gaiaedr3_list.drop(nonmatch_ind,inplace=True)
    cross_match_sum.drop(nonmatch_ind,inplace=True)
    # Re-initialize the indices in the dataframes
    input_gaiaedr3_list.reset_index(inplace=True)
    cross_match_sum.reset_index(inplace=True)
    
    
    # Include basic information
    cross_match_sum['source_id'] = input_gaiaedr3_list['source_id']
    cross_match_sum['ra'] = input_gaiaedr3_list['ra']
    cross_match_sum['dec'] = input_gaiaedr3_list['dec']
    
    # Redefine number of stars from this point on to the number of stars
    # with matched RV
    N_stars = len(input_gaiaedr3_list['source_id'])
    
    print('Starting coordinate transformation for ',N_stars,'stars to galactocentric coordinates...')

    # Calculate the galactocentric coordinate from the input list
    
    input_gaiaedr3_list["XGC"], input_gaiaedr3_list["YGC"], input_gaiaedr3_list["ZGC"] = celes_to_gc_coord_astropy( ra=input_gaiaedr3_list["ra"].values, dec=input_gaiaedr3_list["dec"].values, plx=input_gaiaedr3_list["parallax"].values)

    # Return the crossmatched result
    return input_gaiaedr3_list, cross_match_sum

# Process the Gaia EDR3 dataset

# Time the processing
start_time_p = timeit.default_timer()
cmstar_gaiae3_df, cross_match_sum = cross_match_datasets(allstar_gaiae3_df)
elapsed_p = timeit.default_timer() - start_time_p
print('Total time took to crossmatch:',elapsed_p,"seconds.")


# output result
# run only once
output_path_dr3 = sys.argv[4]
output_path_sum = sys.argv[5]
start_time_w = timeit.default_timer()
cmstar_gaiae3_df.to_csv(output_path_dr3,index=False)
elapsed_w = timeit.default_timer() - start_time_w
print('Time took to write crossmatched results:',elapsed_w,"seconds.")

start_time_w = timeit.default_timer()
cross_match_sum.to_csv(output_path_sum,index=False)
elapsed_w = timeit.default_timer() - start_time_w
print('Time took to write summary:',elapsed_w,"seconds.")


# Examine the crossmatch_summary for metallicity unique matches
series_fe_h = cross_match_sum['LAMOSTDR6LRSSTELLAR']+cross_match_sum['APOGEE']+cross_match_sum['GALAHDR3']+cross_match_sum['RAVEDR6']
count_unique_fe_h = len(np.where(series_fe_h.values != 0)[0])
print('There are',count_unique_fe_h,'unique stars with metallicity measurements from one of the four surveys.')
