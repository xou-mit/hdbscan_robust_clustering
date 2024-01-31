# Xiaowei Ou, May 05, 2021, MIT, 
# Trying to integrate orbits with GAIA EDR3 stars
# with velocities and positions calculated 
# based on zero-point corrected parallaxes and crossmatched RV

#Import the packages
import sys
import os
import timeit
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
coord.galactocentric_frame_defaults.set('v4.0')
import pandas as pd

import warnings
warnings.filterwarnings("once")
#SettingWithCopyWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gala.potential as gp
    import gala.dynamics as gd
    from gala.units import galactic
    import gc

pot = gp.MilkyWayPotential()

from tqdm import tqdm
from multiprocessing import Pool



# Input gaia edr3 cross matched file path
input_path = sys.argv[1]
ind_ini = int(sys.argv[2])
ind_fin = int(sys.argv[3])
assert (ind_fin > ind_ini) & (ind_ini >= 0),"Initial and final indices not acceptable!"

# Import data
cm_vel_pos = pd.read_csv(input_path,skiprows=range(1,ind_ini+1),nrows=ind_fin-ind_ini)

# Check the size
MB = 1024*1024
print("Imported data: %d MB " % (sys.getsizeof(cm_vel_pos)/MB))

output_dir = sys.argv[4]

start_time = timeit.default_timer()

def int_orbit(i):
    
    if np.isnan(XGC_tmp[i]) == True or np.isnan(best_U_tmp[i]) == True:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    XGC = XGC_tmp[i]
    YGC = YGC_tmp[i]
    ZGC = ZGC_tmp[i]
    best_U = best_U_tmp[i]
    best_V = best_V_tmp[i]
    best_W = best_W_tmp[i]
    
    
    samples = coord.SkyCoord(x=XGC*u.pc, y=YGC*u.pc,z=ZGC*u.pc,
    v_x=best_U*u.km/u.second,v_y=best_V*u.km/u.second,v_z=best_W*u.km/u.second,frame=coord.Galactocentric)

    
    # Integrate the orbit
    w0_samples = gd.PhaseSpacePosition(samples.data)
    orbit_samples = gp.hamiltonian.Hamiltonian(pot).integrate_orbit(w0_samples, dt=-1*u.Myr, n_steps=1000)

    start_time = timeit.default_timer()
    # Obtain the orbital properties
    E = orbit_samples.energy().to_value(u.km**2/u.second**2)[0]
    KE = orbit_samples.kinetic_energy().to_value(u.km**2/u.second**2)[0]
    PE = orbit_samples.potential_energy().to_value(u.km**2/u.second**2)[0]
    Lx = orbit_samples.angular_momentum()[0].to_value(u.km*u.kpc/u.second)[0]
    Ly = orbit_samples.angular_momentum()[1].to_value(u.km*u.kpc/u.second)[0]
    Lz = orbit_samples.angular_momentum()[2].to_value(u.km*u.kpc/u.second)[0]
    ecc = orbit_samples.eccentricity()
    zmax = orbit_samples.zmax().to_value(u.pc)
    apo = orbit_samples.apocenter().to_value(u.pc)
    peri = orbit_samples.pericenter().to_value(u.pc)
    
    # Check if the circulation axis is z
    # Flag the result if it's not
    if orbit_samples.circulation()[0] == 0 and orbit_samples.circulation()[1] == 0 and orbit_samples.circulation()[2] == 1:
        flag_circ = 0
    else:
        flag_circ = 1

    # Obtain the action-angle space properties
    try:
        toy_potential = gd.fit_isochrone(orbit_samples)
        flag_fail = 0
        try:
            flag_unbound = 0
            result = gd.find_actions(orbit_samples, N_max=8, toy_potential=toy_potential)
            JR,Jphi,Jz = result['actions'][0].to_value(u.km/u.s*u.kpc), result['actions'][1].to_value(u.km/u.s*u.kpc), result['actions'][2].to_value(u.km/u.s*u.kpc)
        except ValueError:
            flag_unbound = 1
            JR,Jphi,Jz = np.nan, np.nan, np.nan
    except:
        flag_fail = 1
        flag_unbound = np.nan
        JR,Jphi,Jz = np.nan, np.nan, np.nan

    # Check convergence on action and flag if not converged to within 5%
    if abs(Jphi-Lz)/abs(Lz) < 0.05:
        flag_act_conv = 0
    else:
        flag_act_conv = 1

#   The time related functions act strangely within pool... leave it for now
#     if i % 100000 == 0:
#         elapsed = timeit.default_timer() - start_time
#         print("Orbital parameters evaluated up to star", i,"in",elapsed,"seconds.")
    
    return [E, KE, PE, Lx, Ly, Lz, ecc, zmax, apo, peri, JR, Jphi, Jz, flag_circ, flag_act_conv, flag_unbound, flag_fail]

N_stars = len(cm_vel_pos['source_id'])
# Do checkpoints every 50000 stars
cp_every = 50000
N_cp = int(N_stars/cp_every)

# Preextract the numbers in the DataFrame to reduce the overall access number
source_id_tmp = cm_vel_pos["source_id"].values
XGC_tmp = cm_vel_pos["XGC"].values
YGC_tmp = cm_vel_pos["YGC"].values
ZGC_tmp = cm_vel_pos["ZGC"].values
best_U_tmp = cm_vel_pos["best_U"].values
best_V_tmp = cm_vel_pos["best_V"].values
best_W_tmp = cm_vel_pos["best_W"].values

# Delete the extracted columns
del(cm_vel_pos)

# cm_vel_pos.drop(columns=['XGC','YGC','ZGC','best_U','best_V','best_W'],inplace=True)
# print("Data after dropping: %d MB " % (sys.getsizeof(cm_vel_pos)/MB))

# Create the new dataframe for outputing checkpoints
output_df = pd.DataFrame({'source_id': [np.nan]*cp_every, 'Etot': [np.nan]*cp_every, 'KE': [np.nan]*cp_every, 'PE': [np.nan]*cp_every,
'Lx': [np.nan]*cp_every, 'Ly': [np.nan]*cp_every, 'Lz': [np.nan]*cp_every, 'ecc': [np.nan]*cp_every, 'zmax': [np.nan]*cp_every,
'apo': [np.nan]*cp_every, 'peri': [np.nan]*cp_every, 'JR': [np.nan]*cp_every, 'Jphi': [np.nan]*cp_every, 'Jz': [np.nan]*cp_every,
'flag_circ': [np.nan]*cp_every, 'flag_act_conv': [np.nan]*cp_every, 'flag_unbound': [np.nan]*cp_every, 'flag_fail': [np.nan]*cp_every})

# set up a pool of workers
print('Setting up the pool workers now!')
pool = Pool(200)

# Loop through cp_every stars and then checkpoint
for i in range(N_cp+1):
    print('Starting orbit integration (pool) for Checkpoint',i)
    # Determine if we are at the last checkpoint with fewer than cp_every stars left
    if i == N_cp:
        inp = range(i*cp_every,N_stars)
        cp_last = N_stars - i*cp_every
        output_df = pd.DataFrame({'source_id': [np.nan]*cp_last, 'Etot': [np.nan]*cp_last, 'KE': [np.nan]*cp_last, 'PE': [np.nan]*cp_last,
'Lx': [np.nan]*cp_last, 'Ly': [np.nan]*cp_last, 'Lz': [np.nan]*cp_last, 'ecc': [np.nan]*cp_last, 'zmax': [np.nan]*cp_last,
'apo': [np.nan]*cp_last, 'peri': [np.nan]*cp_last, 'JR': [np.nan]*cp_last, 'Jphi': [np.nan]*cp_last, 'Jz': [np.nan]*cp_last,
'flag_circ': [np.nan]*cp_last, 'flag_act_conv': [np.nan]*cp_last, 'flag_unbound': [np.nan]*cp_last, 'flag_fail': [np.nan]*cp_last})
    else:
        inp = range(i*cp_every,(i+1)*cp_every)
    
    # Determine if there's actually no stars at the last checkpoint
    # which is the case if N_star is integer multiple of cp_every
    if len(inp) == 0:
            break
    
    res = pool.map(int_orbit, inp)

    res = np.array(list(zip(*res)))
    print("Finished pool! Adding data to the dataframe and writing out checkpoint",i)
#     print(res)

    # Store the orbital properties
    output_df['source_id'] = source_id_tmp[inp]
    output_df['Etot'] = res[0]
    output_df['KE'] = res[1]
    output_df['PE'] = res[2]
    output_df['Lx'] = res[3]
    output_df['Ly'] = res[4]
    output_df['Lz'] = res[5]
    output_df['ecc'] = res[6]
    output_df['zmax'] = res[7]
    output_df['apo'] = res[8]
    output_df['peri'] = res[9]
    output_df['JR'] = res[10]
    output_df['Jphi'] = res[11]
    output_df['Jz'] = res[12]
    output_df['flag_circ'] = res[13]
    output_df['flag_act_conv'] = res[14]
    output_df['flag_unbound'] = res[15]
    output_df['flag_fail'] = res[16]

    # output result
    # run only once
    print('Outputting the orbital properties checkpoint',i)
    output_path = output_dir + 'orb_param_checkpoint_' + str(i) + '.csv'
    start_time_w = timeit.default_timer()
    output_df.to_csv(output_path,index=False)
    elapsed_w = timeit.default_timer() - start_time_w
    print('Time took to write results:',elapsed_w,"seconds.")

# Close the pool
pool.close()
pool.join()