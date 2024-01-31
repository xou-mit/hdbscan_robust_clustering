# Import the libraries and the datasets
print('Loading libraries...')
import numpy as np
import uncertainties
from uncertainties import unumpy
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
from multiprocessing import Pool
import gc

print('Loading datasets...')
# Input gaia edr3 cross matched file path
input_path = sys.argv[1]

cmstar_df = pd.read_csv(input_path)
# cmstar_df = np.genfromtxt(input_path,delimiter=',',names=True,
#                           dtype='i8,i8'+',f8'*13+',i'+',f8'*6+',i'+',f8'*2+',i'+',f8'*4+',i'+',f8'*2+',i'+',f8'*4
#                  +',i'+',f8'*2+',i'+',f8'*17)

MB = 1024*1024
print("Imported data: %d MB " % (sys.getsizeof(cmstar_df)/MB))

# Dissect the array if needed for testing
# Do the whole DF if both arguments are the same
ind_ini = int(sys.argv[2])
ind_fin = int(sys.argv[3])
if ind_ini < ind_fin:
    cmstar_df = cmstar_df[ind_ini:ind_fin]

output_dir = sys.argv[4]
    
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

T_mat1 = np.array([[cos(theta),sin(theta),0],[sin(theta),-cos(theta),0],[0,0,1]])
T_mat2 = np.array([[-sin(dec_NGP),0,cos(dec_NGP)],[0,1,0],[cos(dec_NGP),0,sin(dec_NGP)]])
T_mat3 = np.array([[cos(ra_NGP),sin(ra_NGP),0],[-sin(ra_NGP),cos(ra_NGP),0],[0,0,1]])
T_mat = T_mat1@T_mat2@T_mat3

def rv_pm_to_gc_vel(tup,if_forder=False):
    '''
    Conversion from radial velocities and proper motions 
    to velocities in heliocentric galactic rectangular coordinate
    with unumpy or first order error propagation.
    Assumed the input ra and dec are in degrees
    parallex and its error are in milli-arcsec
    all proper motions and their errors are in 
    milli-arcsec per year.
    Also assume the input pmra has already be multiplied by cos(dec)
    The input cov numbers are actually correlation
    
    This version is specialized to take only one input for multiprocessing
    '''
    # Unpack the tuple into variables
    i, j = tup
    
    # Check if there's valid RV measurements
#     if np.isnan(z_tmp[j][i]) == True:
    if np.isnan(cmstar_df.loc[i,z_list[j]]) == True:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0]       
    
    ra = ra_tmp[i]
    dec = dec_tmp[i]
    plx = plx_tmp[i]
    plxerr = plxerr_tmp[i]
    z = cmstar_df.loc[i,z_list[j]]
    zerr = cmstar_df.loc[i,zerr_list[j]]
    pmra = pmra_tmp[i]
    pmraerr = pmraerr_tmp[i]
    pmdec = pmdec_tmp[i]
    pmdecerr = pmdecerr_tmp[i]
    plxpmracov = plxpmracorr_tmp[i]
    plxpmdeccov = plxpmdeccorr_tmp[i]
    pmrapmdeccov = pmrapmdeccorr_tmp[i]
    
    # First calculate the velocity itself
    # Make the unit conversion to match the formulae shown in Jo Bovy's note
    ra,dec = ra*np.pi/180, dec*np.pi/180
    pmra, pmdec, plx = pmra/1000, pmdec/1000, plx/1000
    pmraerr, pmdecerr, plxerr = pmraerr/1000, pmdecerr/1000, plxerr/1000
    
    RV = c*z
    RVerr = c*zerr
    
    # Calculate the A matrix, which depends on ra and dec
    A_mat1 = np.array([[cos(ra),-sin(ra),0],[sin(ra),cos(ra),0],[0,0,1]])
    A_mat2 = np.array([[cos(dec),0,-sin(dec)],[0,1,0],[sin(dec),0,cos(dec)]])
    A_mat = A_mat1@A_mat2
    
    corr_matrix = np.array([[ 1.          ,  pmrapmdeccov,  plxpmracov ],
                            [ pmrapmdeccov,  1.          ,  plxpmdeccov],
                            [ plxpmracov  ,  plxpmdeccov ,  1.         ]])
    if if_forder == False:
        try:
            (pmra_u, pmdec_u, plx_u) = uncertainties.correlated_values_norm([(pmra,pmraerr), (pmdec,pmdecerr), (plx,plxerr)], corr_matrix)
            RV_u = uncertainties.ufloat(RV, RVerr)
            d_u = 1/plx_u

            UVW = T_mat@A_mat@np.array([[RV_u],[k*d_u*pmra_u],[k*d_u*pmdec_u]])

            # Get the covariance matrix
            UVW_cov = uncertainties.covariance_matrix([UVW[0,0], UVW[1,0], UVW[2,0]])
        except: #np.linalg.LinAlgError:
            # Use Lina's first order approximation for error
            dis = 1/plx
            dis_s = abs(plxerr/plx**2)

            corr_mu_mu = pmrapmdeccov*pmraerr*pmdecerr
            corr_d_alpha = plxpmracov*dis_s*pmraerr 
            corr_d_delta = plxpmdeccov*dis_s*pmdecerr

            covariance_d_alpha_delta = np.array([[dis_s**2, corr_d_alpha, corr_d_delta ],[corr_d_alpha, pmraerr**2, corr_mu_mu],[corr_d_delta, corr_mu_mu, pmdecerr**2] ])
            bmatrix = k*np.array([[pmra, dis, 0],[pmdec, 0, dis]])

            covariance_valpha_vdelta = np.dot(np.dot(bmatrix, covariance_d_alpha_delta), np.transpose(bmatrix))

            covariance_vr_valpha_vdelta = np.array([[RVerr**2,0,0],[0,covariance_valpha_vdelta[0][0], covariance_valpha_vdelta[0][1] ],[0,covariance_valpha_vdelta[1][0], covariance_valpha_vdelta[1][1]]])
            TA_mat = T_mat@A_mat

            UVW = T_mat@A_mat@np.array([[RV],[k*dis*pmra],[k*dis*pmdec]])
            UVW_cov = TA_mat@covariance_vr_valpha_vdelta@TA_mat.T
    else:
        # Use Lina's first order approximation directly if flag is set
        dis = 1/plx
        dis_s = abs(plxerr/plx**2)

        corr_mu_mu = pmrapmdeccov*pmraerr*pmdecerr
        corr_d_alpha = plxpmracov*dis_s*pmraerr 
        corr_d_delta = plxpmdeccov*dis_s*pmdecerr

        covariance_d_alpha_delta = np.array([[dis_s**2, corr_d_alpha, corr_d_delta ],[corr_d_alpha, pmraerr**2, corr_mu_mu],[corr_d_delta, corr_mu_mu, pmdecerr**2] ])
        bmatrix = k*np.array([[pmra, dis, 0],[pmdec, 0, dis]])

        covariance_valpha_vdelta = np.dot(np.dot(bmatrix, covariance_d_alpha_delta), np.transpose(bmatrix))

        covariance_vr_valpha_vdelta = np.array([[RVerr**2,0,0],[0,covariance_valpha_vdelta[0][0], covariance_valpha_vdelta[0][1] ],[0,covariance_valpha_vdelta[1][0], covariance_valpha_vdelta[1][1]]])
        TA_mat = T_mat@A_mat

        UVW = T_mat@A_mat@np.array([[RV],[k*dis*pmra],[k*dis*pmdec]])
        UVW_cov = TA_mat@covariance_vr_valpha_vdelta@TA_mat.T
    
    
    # Correct for the solar velocity
    UVW = UVW + np.array([[12.9],[245.6],[7.78]])
    
    # Assign the values to the output arrays
    if if_forder == False:
        try:
            # In case the first order approximation was made
            U = UVW[0,0].n
            V = UVW[1,0].n
            W = UVW[2,0].n
            Uerr = UVW[0,0].s
            Verr = UVW[1,0].s
            Werr = UVW[2,0].s
            v_forder_flag = 0
        except AttributeError:
            U = UVW[0,0]
            V = UVW[1,0]
            W = UVW[2,0]
            Uerr = UVW_cov[0,0]**(1/2)
            Verr = UVW_cov[1,1]**(1/2)
            Werr = UVW_cov[2,2]**(1/2)
            v_forder_flag = 1
    else:
        U = UVW[0,0]
        V = UVW[1,0]
        W = UVW[2,0]
        Uerr = UVW_cov[0,0]**(1/2)
        Verr = UVW_cov[1,1]**(1/2)
        Werr = UVW_cov[2,2]**(1/2)
        v_forder_flag = 1
    
    UV_cov, UW_cov, VW_cov = UVW_cov[0][1], UVW_cov[0][2], UVW_cov[1][2]
    
    
    return [U, V, W, Uerr, Verr, Werr, UV_cov, UW_cov, VW_cov, v_forder_flag]

def get_vel_cyl(**kwargs):
    # Create the astropy sample
    samples = coord.SkyCoord(**kwargs)
    vtot = np.sqrt(samples.v_x.to_value(u.km/u.s)**2+samples.v_y.to_value(u.km/u.s)**2+samples.v_z.to_value(u.km/u.s)**2)
    samples.representation_type = 'cylindrical'
    
    return samples.d_rho.to_value(u.km/u.s), samples.d_phi.to_value(u.rad/u.s)*samples.rho.to_value(u.km), samples.d_z.to_value(u.km/u.s)


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

N_datasets = len(dataset_list)
N_stars = len(cmstar_df['source_id'])

print('Starting coordinate transformation for ',N_stars,'stars to galactocentric coordinates...')

# Time the velocity conversion
start_time_v = timeit.default_timer()

# Calculate the velocities from these datasets
# Preextract the numbers in the DataFrame to reduce the overall access number
ra_tmp = cmstar_df["ra"].values
dec_tmp = cmstar_df["dec"].values
plx_tmp = cmstar_df["parallax"].values
plxerr_tmp = cmstar_df["parallax_error"].values
pmra_tmp = cmstar_df["pmra"].values
pmraerr_tmp = cmstar_df["pmra_error"].values
pmdec_tmp = cmstar_df["pmdec"].values
pmdecerr_tmp = cmstar_df["pmdec_error"].values
plxpmracorr_tmp = cmstar_df["parallax_pmra_corr"].values
plxpmdeccorr_tmp = cmstar_df["parallax_pmdec_corr"].values
pmrapmdeccorr_tmp = cmstar_df["pmra_pmdec_corr"].values
xgc_tmp = cmstar_df['XGC'].values
ygc_tmp = cmstar_df['YGC'].values
zgc_tmp = cmstar_df['ZGC'].values

# Preextract the numbers in the DataFrame to reduce the overall access number
# z_tmp = [cmstar_df[z_name].values for z_name in z_list]
# zerr_tmp = [cmstar_df[zerr_name].values for zerr_name in zerr_list]
# zqul_tmp = [cmstar_df[zqul_name].values for zqul_name in zqul_list]
# z_tmp = [cmstar_df[z_name] for z_name in z_list]
# zerr_tmp = [cmstar_df[zerr_name] for zerr_name in zerr_list]

# Drop the extracted columns
cmstar_df.drop(columns=['ra','dec','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
                       'parallax_pmra_corr','parallax_pmdec_corr','pmra_pmdec_corr','XGC','YGC','ZGC',
                       'dr2_radial_velocity','dr2_radial_velocity_error','dr2_rv_nb_transits',
                       'm_h_ap','m_h_err_ap','m_h_flg_ap','alpha_m_ap','alpha_m_err_ap','alpha_m_flg_ap',
                       'm_h_l6s','m_h_err_l6s','m_h_flg_l6s','alpha_m_l6s','alpha_m_err_l6s',
                       'm_h_gl3','m_h_err_gl3','m_h_flg_gl3','alpha_m_gl3','alpha_m_err_gl3','alpha_m_flg_gl3',
                       'm_h_r6','m_h_err_r6','alpha_m_r6','alpha_m_err_r6'],inplace=True)

print("Data after dropping: %d MB " % (sys.getsizeof(cmstar_df)/MB))

# set up a pool of workers
print('Setting up the pool workers now!')
pool = Pool(80)

# Create the array to store the best values
cmstar_df['best_dataset'] = np.nan 
cmstar_df['best_z'] = np.nan
cmstar_df['best_zerr'] = np.nan
cmstar_df['best_zqul'] = np.nan
cmstar_df['best_U'] = np.nan
cmstar_df['best_V'] = np.nan
cmstar_df['best_W'] = np.nan
cmstar_df['best_Uerr'] = np.nan
cmstar_df['best_Verr'] = np.nan
cmstar_df['best_Werr'] = np.nan
cmstar_df['best_UVcov'] = np.nan
cmstar_df['best_UWcov'] = np.nan
cmstar_df['best_VWcov'] = np.nan
cmstar_df['best_v_forder_flag'] = 0

# Initialize an array to store the current best zqul
# set to arb. large value
prev_zqul = np.ones(N_stars)*99


for j in range(N_datasets):    
    z_name = z_list[j]
    zerr_name = zerr_list[j]
    U_name, V_name, W_name = U_list[j], V_list[j], W_list[j]
    Uerr_name, Verr_name, Werr_name = Uerr_list[j], Verr_list[j], Werr_list[j]
    UVcov_name, UWcov_name, VWcov_name = UVcov_list[j], UWcov_list[j], VWcov_list[j]
    v_forder_flag_name = v_forder_flag_list[j]
        
        
    print('Starting velocity calculation (pool) on dataset:',dataset_list[j])
    inp = list(zip(range(N_stars),[j]*len(range(N_stars))))
    res = pool.map(rv_pm_to_gc_vel, inp)
    
    res = list(zip(*res))
    print("Finished pool! Adding data to the dataframe and writing out checkpoint!")
    
    cmstar_df[U_name],cmstar_df[Uerr_name] = res[0], res[3]
    cmstar_df[V_name],cmstar_df[Verr_name] = res[1], res[4]
    cmstar_df[W_name],cmstar_df[Werr_name] = res[2], res[5]
    cmstar_df[UVcov_name], cmstar_df[UWcov_name], cmstar_df[VWcov_name] = res[6], res[7], res[8]
    cmstar_df[v_forder_flag_name] = res[9]
    
    
    
    print("Writing checkpoint...")
    start_time_c = timeit.default_timer()
    checkpoint_path = output_dir + dataset_list[j] + '.csv'
    cmstar_df.to_csv(checkpoint_path,columns=['index',U_name, Uerr_name, V_name, Verr_name, W_name, Werr_name, UVcov_name, UWcov_name, VWcov_name, v_forder_flag_name],index=False)
    elapsed_c = timeit.default_timer() - start_time_c
    print('Time took to write checkpoint:',elapsed_c,"seconds.")
    
    # Compare the current zqul with the previous zqul
    current_zqul = cmstar_df.loc[:,zqul_list[j]]
    zqul_diff = current_zqul - prev_zqul
    ind_better = np.where((zqul_diff < 0) | ((np.isnan(prev_zqul) == True) & (np.isnan(current_zqul) == False)))[0]
    # Update the best values 
    cmstar_df.loc[ind_better,best_col] = cmstar_df.loc[ind_better,col_arr[:,j]].values
    cmstar_df.loc[ind_better,'best_dataset'] = dataset_list[j]
    prev_zqul = cmstar_df.loc[:,'best_zqul'].values

print('Finished velocity transformation and closing the pool...')

pool.close()
pool.join()

elapsed_v = timeit.default_timer() - start_time_v
print('Total time took to convert velocities:',elapsed_v,"seconds.")

# Get extra columns of the cylindrical velocities of the stars
cmstar_df['best_vr'], cmstar_df['best_vphi'], cmstar_df['best_vz'] = get_vel_cyl(x=xgc_tmp*u.pc, y=ygc_tmp*u.pc, z=zgc_tmp*u.pc, v_x=cmstar_df['best_U'].values*u.km/u.second, v_y=cmstar_df['best_V'].values*u.km/u.second, v_z=cmstar_df['best_W'].values*u.km/u.second, frame=coord.Galactocentric)
print('Converted the best velocities to cylindrical coordinate.')

# output result
# run only once
print('Outputting the best velocities...')
output_path = output_dir + 'best_for_' + str(N_stars) + '_stars.csv'
start_time_w = timeit.default_timer()
cmstar_df.to_csv(output_path,columns=['index','source_id','best_dataset','best_vr','best_vphi','best_vz']+best_col,index=False)
elapsed_w = timeit.default_timer() - start_time_w
print('Time took to write results:',elapsed_w,"seconds.")
