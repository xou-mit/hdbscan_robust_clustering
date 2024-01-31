# XO note on 04/11/2022: This is the script for running HDBSCAN clustering code on input values sampled from their 
# corresponding uncertainties
# Note even though this script allows PCA, it will not work with error sampling since the sampling is done before
# adding in additional axes. Thus, the added axes will not be resampled and the output clusters mean will not include
# those axes.
# Do not use PCA!
# XO note on 02/10/2020: I added an additional cut on the relative errors of the input parameters
# to improve the stability of the resulting clusters. 
# XO note on 04/11/2022: I modified the input so that the clustering will be done on a sample of specidfied list of indices of stars
# Will only make plots for the first 20 realizations to save time


# Expected input: python filename.py input_ind_date_file N_stack N_samp min_samples min_cluster_size data_dir [plots_dir]

# Import libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

cmap = ListedColormap(sns.color_palette("colorblind",256))


import sklearn
# print(sklearn.__version__)
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import OPTICS
# from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import hdbscan

import warnings
warnings.filterwarnings("ignore")
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
print("Finished importing libraries...")

# Define a plotting funciton for plotting specific lists of stars in all relevant spaces
def plot_summary_v3(df_vel,df_orb,ind_list,c_list,title,legends,list_mask=False,filename=None,bg=False,ds=True,noise=True,ind_add_cut=[]):
    if len(ind_add_cut) == 0 and list_mask == True:
        ind_add_cut = ind_cut
    
    f = plt.figure(figsize=[30,40])
    
#     plt.suptitle(title)
    
    # Set the plotting parameters
    plt.subplot(4,3,1)
    plt.xlim([-4.5e3,4.5e3])
    plt.ylim([-1.7e5,0.1e5])
    plt.xlabel(r'$L_z$ [kpc km s$^{-1}$]')
    plt.ylabel(r'$E_{tot}$ [km$^2$ s$^{-2}$]')
    
    # Make a fake dot for the legend
    # Adjusted the auto-legend setting to always make stars for clusters and dots for noises
    for i in range(len(ind_list)):
        if ((i < len(ind_list)-1 or noise == False) or (len(ind_list) == 1)) and bg == False:  
            mkr='*'
            ec='k'
        else:
            mkr='.'
            ec=c_list[i][0]
        try:
            # Check if needs downsampling; if so, flag it in legend
            if len(ind_list[i]) > 5000 and ds == True:
                plt.scatter([],[],s=100,alpha=0.8,c=c_list[i][0],label=legends[i]+'_sp',marker=mkr,edgecolors=ec)
            else:
                plt.scatter([],[],s=100,alpha=0.8,c=c_list[i][0],label=legends[i],marker=mkr,edgecolors=ec)
        except IndexError:
            print("Cluster number:",legends[i],'has zero stars.')
#     plt.legend(fontsize='small')
    
    plt.subplot(4,3,2)
    plt.xlim([3000,13000])
    plt.ylim([-5000,5000])
    plt.xlabel(r'$r_{gal}$ [pc]')
    plt.ylabel('Z [pc]')
    
    plt.subplot(4,3,3)
    plt.xlim([0,26000])
    plt.ylim([0,26000])
    plt.xlabel(r'$r_{peri}$ [pc]')
    plt.ylabel(r'$r_{apo}$ [pc]')
    
    plt.subplot(4,3,4)
    plt.xlim([-600,600])
    plt.ylim([-600,600])
    plt.xlabel(r'$v_{r}$ [km/s]')
    plt.ylabel(r'$v_{\phi}$ [km/s]')
    
    plt.subplot(4,3,5)
    plt.xlim([-600,600])
    plt.ylim([-600,600])
    plt.xlabel(r'$v_{r}$ [km/s]')
    plt.ylabel(r'$v_{z}$ [km/s]')
    
    plt.subplot(4,3,6)
    plt.xlim([0,1])
    plt.xlabel(r'Ecc [Unitless]')
    plt.ylabel(r'$f$(Ecc)')
    
    plt.subplot(4,3,7)
    plt.xlim([0,1])
    plt.ylim([1000,5000])
    plt.xlabel('Ecc [unitless]')
    plt.ylabel('Zmax [pc]')
    
    plt.subplot(4,3,8)
    plt.xlabel(r'$J_{\phi}/J_{tot}$ [kpc km/s]')
    plt.ylabel(r'$(J_z-J_R)/J_{tot}$ [kpc km/s]')
    
    plt.subplot(4,3,9)
    plt.xlim([-3.1,1.0])
    plt.ylim([-0.21,0.61])
    plt.xlabel('[Fe/H] [dex]')
    plt.ylabel(r'[$\alpha$/Fe] [dex]')
    
    plt.subplot(4,3,10)
    plt.xlim([0, 9500])
    plt.ylim([-4500, 4500])
    plt.xlabel(r'$J_R$ [kpc km/s]')
    plt.ylabel(r'$J_{\phi}$ [kpc km/s]')
    
    plt.subplot(4,3,11)
    plt.xlim([0, 9500])
    plt.ylim([0, 4500])
    plt.xlabel(r'$J_R$ [kpc km/s]')
    plt.ylabel(r'$J_z$ [kpc km/s]')
    
    plt.subplot(4,3,12)
    plt.xlim([-3.1,1.0])
    plt.xlabel(r'[Fe/H] [dex]')
    plt.ylabel(r'$f$([Fe/H])')
    
    for i in range(len(ind_list)):
        if list_mask == True:
            ind = ind_add_cut[ind_list[i]]
        else:
            ind = ind_list[i]
        color = c_list[i]
        
            
        if ((i < len(ind_list)-1 or noise == False) or (len(ind_list) == 1)) and bg == False:
            mkr='*'
            ms=50
            al=1 
#             ec=['k']*len(ind)
            ec='none'
            zord=i
        else:
            mkr='.'
            ms=20
            al=0.3
            ec='none'
            zord=-1
        
        # Determine if there are more than 10000 data points, if so, down-sample it
        if len(ind) > 5000 and ds == True:
            ii = np.random.choice(len(ind),int(len(ind)/20),replace=False)
            ind = ind[ii]
            
            # Check to see if color list also needs to be downsampled
            if len(color) > len(ind): 
                color = color[ii]
        
        # Plot E vs. Lz
        plt.subplot(4,3,1)
        plt.scatter(df_orb.loc[ind,'Lz'],df_orb.loc[ind,'Etot'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        # Plot Z vs. r_gal
        plt.subplot(4,3,2)
        plt.scatter(np.sqrt(df_vel.loc[ind,'XGC']**2+df_vel.loc[ind,'YGC']**2),df_vel.loc[ind,'ZGC'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        # Plot apo vs. peri
        plt.subplot(4,3,3)
        plt.scatter(df_orb.loc[ind,'peri'],df_orb.loc[ind,'apo'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        
        # Plot vphi vs. vr
        plt.subplot(4,3,4)
        plt.scatter(df_vel.loc[ind,'vr_g2'],df_vel.loc[ind,'vphi_g2'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        
        # Plot vz vs. vr
        plt.subplot(4,3,5)
        plt.scatter(df_vel.loc[ind,'vr_g2'],df_vel.loc[ind,'vz_g2'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        
        # Plot 1d Ecc histogram: skip the noise
        if ((i < len(ind_list)-1 or noise == False) or (len(ind_list) == 1)):
            plt.subplot(4,3,6)
#             plt.hist(df_orb.loc[ind,'ecc'],histtype='step',bins=np.arange(0.0,1.05,0.05),color=color[0])
            sns.distplot(df_orb.loc[ind,'ecc'],hist=False,kde=True,bins=np.arange(0.0,1.05,0.05),kde_kws={"color": color[0]})
            # Have to set the x-axis again to overwrite sns auto axis
            plt.xlabel(r'Ecc [Unitless]')
        
        
        # Plot zmax vs. ecc
        plt.subplot(4,3,7)
        plt.scatter(df_orb.loc[ind,'ecc'],df_orb.loc[ind,'zmax'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
    
        # Plot Action diamond
        plt.subplot(4,3,8)
        plt.scatter(df_orb.loc[ind,'diamond_x'],df_orb.loc[ind,'diamond_y'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        
        # Plot Alpha vs. [Fe/H]
        plt.subplot(4,3,9)
        plt.scatter(df_vel.loc[ind,'m_h_ap17'],df_vel.loc[ind,'alpha_m_ap17'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        
        # Plot Jphi vs. JR
        plt.subplot(4,3,10)
        plt.scatter(df_orb.loc[ind,'JR'],df_orb.loc[ind,'Jphi'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        
        # Plot Jz vs. JR
        plt.subplot(4,3,11)
        plt.scatter(df_orb.loc[ind,'JR'],df_orb.loc[ind,'Jz'],s=ms,alpha=al,c=color,edgecolors=ec,linewidths=1,marker=mkr,zorder=zord)
        
        
        # Plot 1d [Fe/H] histogram; skip the noise
        if ((i < len(ind_list)-1 or noise == False) or (len(ind_list) == 1)):
            plt.subplot(4,3,12)
#             plt.hist(df_vel.loc[ind,'m_h_mean'],histtype='step',bins=np.arange(-3.1,1.1,0.1),color=color[0])
            sns.distplot(df_vel.loc[ind,'m_h_mean'],hist=False,kde=True,bins=np.arange(-3.1,1.1,0.1),kde_kws={"color": color[0]})
            plt.xlabel(r'[Fe/H] [dex]')
        
    
    f.tight_layout()
    if filename != None:
        f.savefig(filename,layout='tight')
    plt.close()

# Import data
cm_vel_all = pd.read_hdf('../data/dr3_near_vel_plxzp_g2_only.h5')
orb_param_all = pd.read_hdf('../data/dr3_orb_param_err_g2_only.h5')

print("Finished importing data...")

# calculate the columns for the scaled action diamond
orb_param_all['Jtot'] = np.sqrt(orb_param_all['Jphi']**2+orb_param_all['JR']**2+orb_param_all['Jz']**2)
orb_param_all['diamond_x']=orb_param_all['Jphi']/orb_param_all['Jtot']
orb_param_all['diamond_y']=(orb_param_all['Jz']-orb_param_all['JR'])/orb_param_all['Jtot']

# Calculate the error for Jtot and diamond_x/y
orb_param_all['e_Jtot'] = np.sqrt(orb_param_all['Jphi']**2*orb_param_all['e_Jphi']**2+orb_param_all['JR']**2*orb_param_all['e_JR']**2+orb_param_all['Jz']**2*orb_param_all['e_Jz']**2)/orb_param_all['Jtot']
orb_param_all['e_diamond_x']=np.sqrt(orb_param_all['Jtot']**2*orb_param_all['e_Jphi']**2+orb_param_all['e_Jtot']**2*orb_param_all['Jphi']**2)/orb_param_all['Jtot']**2
orb_param_all['e_diamond_y']=np.sqrt(orb_param_all['Jtot']**2*(orb_param_all['e_JR']**2+orb_param_all['e_Jz']**2)+orb_param_all['e_Jtot']**2*(orb_param_all['Jz']-orb_param_all['JR'])**2)/orb_param_all['Jtot']**2

# Calculate L_perp for clustering
orb_param_all['Lperp'] = np.sqrt(orb_param_all['Lx']**2+orb_param_all['Ly']**2)
orb_param_all['e_Lperp'] = np.sqrt(orb_param_all['Lx']**2*orb_param_all['e_Lx']**2+orb_param_all['Ly']**2*orb_param_all['e_Ly']**2)/orb_param_all['Lperp']

# This mean metallicity below is only used in the case when we actually want to cluster in or with a prior cut on metallicity
# To avoid systematic differences between spectroscopic surveys; don't mix metallicities from different surveys
# cm_vel_all['m_h_mean'] = np.nanmean(cm_vel_all[['m_h_ap17','m_h_l6s','m_h_r6c','m_h_gl3']].values, axis=1).T
cm_vel_all['m_h_mean'], cm_vel_all['e_m_h_mean'] = cm_vel_all['m_h_ap17'], cm_vel_all['m_h_err_ap17'] 

orb_param_all['PCA_X'] = np.empty(len(orb_param_all))*np.nan
orb_param_all['PCA_Y'] = np.empty(len(orb_param_all))*np.nan

print("Finished calculating additional axes...")

# Tweak with the axes that go into PCA

scaler = 'Robust' # 'Standard' or 'Robust' or None
denoise = None # 'PCA' or 'AE' or None
algorithm = 'HDBSCAN' # 'OPTICS' or 'DBSCAN' or 'HDBSCAN' or 'AGG_n' or 'AGG_l' or 'GMM'
# Define what axes go into PCA
action = True
diamond = False
metallicity = False
velocity = False
cylindrical = True
position = False
energy = True
eccentricity = False
Lz = False
Lperp = False

# Define what axes go into GMM with PCA results
# By default, include whatever was not in PCA
action_add = not action
metallicity_add = not metallicity
velocity_add = not velocity
position_add = not position
energy_add = not energy
eccentricity_add = not eccentricity
Lz_add = not Lz
Lperp_add = not Lperp

# Manual override additional axes
action_add = False
position_add = False
metallicity_add = False
velocity_add = False
energy_add = False
eccentricity_add = False
Lz_add = False
Lperp_add = False


# Pick the stars from the input index list [xx]
ind_cut = np.load(sys.argv[1],allow_pickle=True)


# Cut the dataframe
df_cut_vel = cm_vel_all.loc[ind_cut,:]
df_cut_orb = orb_param_all.loc[ind_cut,:]

data_dir = sys.argv[6]
# Check if there's an input for plotting directory and set the flag for making plots
try:
    plots_dir = sys.argv[7]
    plotting_flag = True
except:
    plotting_flag = False

extratext = '_0412_Nstackcut_'+str(sys.argv[2])

if denoise == 'PCA':
    if action_add == True:
        extratext += '_act'

    if metallicity_add == True:
        extratext += '_feh'

    if velocity_add == True:
        extratext += '_vel'

    if position_add == True:
        extratext += '_pos'

    if energy_add == True:
        extratext += '_etot'

    if eccentricity_add == True:
        extratext += '_ecc'

    if Lz_add == True:
        extratext += '_Lz'
        
    if Lperp_add == True:
        extratext += '_Lperp'
else:
    if action == True or action_add == True:
        extratext += '_act'
        if diamond == True:
            extratext += '_diamond'

    if metallicity == True or metallicity_add == True:
        extratext += '_feh'

    if velocity == True or velocity_add == True:
        extratext += '_vel'
        if cylindrical == True:
            extratext += '_cyl'

    if position == True or position_add == True:
        extratext += '_pos'

    if energy == True or energy_add == True:
        extratext += '_etot'

    if eccentricity == True or eccentricity_add == True:
        extratext += '_ecc'

    if Lz == True or Lz_add == True:
        extratext += '_Lz'
        
    if Lperp == True or Lperp_add == True:
        extratext += '_Lperp'

# Clustering options
if algorithm == 'DBSCAN':
    eps = 0.1
    extratext += '_eps01'
    min_samples = 10
    extratext += '_min'+str(min_samples)
elif algorithm == 'OPTICS':
    min_samples = 12
    extratext += '_opt'
    extratext += '_min'+str(min_samples)
elif algorithm == 'HDBSCAN':
    extratext += '_hdbscan'
#     cluster_selection_epsilon = 0.
#     extratext += '_eps0'
    min_samples = int(sys.argv[4])
    min_cluster_size = int(sys.argv[5])
    extratext += '_min_samples_'+str(min_samples)
    extratext += '_min_clustsize_'+str(min_cluster_size)
    cluster_selection_method = 'leaf'
    extratext += '_'+str(cluster_selection_method)
elif algorithm == 'AGG_n':
    # Use n_clusters
    extratext += '_agglomerative'
    n_cluster = 3
    linkage_threshold = None
    extratext += '_n_clust_'+str(n_cluster)
elif algorithm == 'AGG_l':
    # Use linkage_threshold
    extratext += '_agglomerative'
    n_cluster = None
    linkage_threshold = 25
    extratext += '_l_threshold_25'
elif algorithm == 'GMM':
    extratext += '_gmm'

# Decomposition options
if denoise == 'PCA':
    extratext += '_PCA'
    n_pca_comp = 2
    extratext += '_ncomp_'+str(n_pca_comp)


# Make a list that records what axes are fed into PCA (or not)
pca_list = []


# Add in the dimensions as needed 
ydata_ini = []
ydata_ini_err = []

if action == True and diamond == True:
    ydata_ini.append(df_cut_orb['diamond_x'])
    ydata_ini.append(df_cut_orb['diamond_y'])
    ydata_ini_err.append(df_cut_orb['e_diamond_x'])
    ydata_ini_err.append(df_cut_orb['e_diamond_y'])
    pca_list.append('Act_diamond')
    
if action == True and diamond == False:
    ydata_ini.append(df_cut_orb['JR'])
    ydata_ini.append(df_cut_orb['Jphi'])
    ydata_ini.append(df_cut_orb['Jz'])
    ydata_ini_err.append(df_cut_orb['e_JR'])
    ydata_ini_err.append(df_cut_orb['e_Jphi'])
    ydata_ini_err.append(df_cut_orb['e_Jz'])
    pca_list.append('Act_3d')

    
if velocity == True and cylindrical == False:
    ydata_ini.append(df_cut_vel["U_g2"])
    ydata_ini.append(df_cut_vel["V_g2"])
    ydata_ini.append(df_cut_vel["W_g2"])
    ydata_ini_err.append(df_cut_vel["Uerr_g2"])
    ydata_ini_err.append(df_cut_vel["Verr_g2"])
    ydata_ini_err.append(df_cut_vel["Werr_g2"])
    pca_list.append('Vel_cart')
    
if metallicity == True:
    ydata_ini.append(df_cut_vel['m_h_mean'])
    ydata_ini_err.append(df_cut_vel['e_m_h_mean'])
    pca_list.append('[Fe/H]')
    
if velocity == True and cylindrical == True:
    ydata_ini.append(df_cut_vel["vr_g2"])
    ydata_ini.append(df_cut_vel["vphi_g2"])
    ydata_ini.append(df_cut_vel["vz_g2"])
    ydata_ini_err.append(df_cut_vel["vrerr_g2"])
    ydata_ini_err.append(df_cut_vel["vphierr_g2"])
    ydata_ini_err.append(df_cut_vel["vzerr_g2"])
    pca_list.append('Vel_cyl')
    
if position == True:
    ydata_ini.append(df_cut_vel["XGC"])
    ydata_ini.append(df_cut_vel["YGC"])
    ydata_ini.append(df_cut_vel["ZGC"])
    ydata_ini_err.append(df_cut_vel["XGCerr"])
    ydata_ini_err.append(df_cut_vel["YGCerr"])
    ydata_ini_err.append(df_cut_vel["ZGCerr"])
    pca_list.append('Pos_cart')

if energy == True:
    ydata_ini.append(df_cut_orb["Etot"])
    ydata_ini_err.append(df_cut_orb["e_Etot"])
    pca_list.append('E_tot')

if eccentricity == True:
    ydata_ini.append(df_cut_orb["ecc"])
    ydata_ini_err.append(df_cut_orb["e_ecc"])
    pca_list.append('ecc')

if Lz == True:
    ydata_ini.append(df_cut_orb["Lz"])
    ydata_ini_err.append(df_cut_orb["e_Lz"])
    pca_list.append('Lz')

if Lperp == True:
    ydata_ini.append(df_cut_orb["Lperp"])
    ydata_ini_err.append(df_cut_orb["e_Lperp"])
    pca_list.append('Lperp')
    


ydata = np.array(ydata_ini).T
ydata_err = np.array(ydata_ini_err).T


# Sample the ydata before feeding into scaler and the clustering algorithm
np.random.seed(42)

# Generate a random list of seed from the single seed specified above
N_samp = int(sys.argv[3])
rand_seed_list = np.random.choice(np.arange(10000,dtype=int),N_samp,replace=False)

# Ask for user confirmation from the extratext
# print("Extra text is:",extratext)
# confirm = input("Continue? (Y/N)")

# if confirm == 'Y':
#     pass
# elif confirm == 'N':
#     sys.exit()


# save the random seeds
np.save(data_dir + 'random_seed' + extratext + '.npy', rand_seed_list)

# Generate the random initial conditions assuming all inputs are independent without covariance
# this is not true since cylindrical velocities are definitely correlated
# Energies/actions are also calculated by sampling velocities, so there must be correlations.
for i in range(N_samp):
    print("Starting trial #",i)
    ydata_samp = np.zeros(np.shape(ydata))
    np.random.seed(rand_seed_list[i])
    
    print("Sampling inputs...")
    for j in range(len(ydata_samp[0,:])):
        ydata_samp[:,j] = np.random.normal(ydata[:,j], ydata_err[:,j])


    print("Scaling inputs...")
    if denoise == None:
        if scaler == 'Standard':
            X = StandardScaler().fit_transform(ydata_samp)
        elif scaler == 'Robust':
            X = RobustScaler().fit_transform(ydata_samp)
        elif scaler == None:
            X = ydata_samp
    elif denoise == 'PCA':
        if scaler == None:
            X = PCA().fit_transform(ydata_samp)
        elif scaler == 'Standard':
            X = StandardScaler().fit_transform(PCA().fit_transform(StandardScaler().fit_transform(ydata_samp)))
        elif scaler == 'Robust':
            X = RobustScaler().fit_transform(PCA().fit_transform(RobustScaler().fit_transform(ydata_samp)))    

    # Print the explained variance ratios
    if denoise == 'PCA':
        print("Axes in PCA:",pca_list)
        print("Explained variance ratios from PCA:",PCA().fit(StandardScaler().fit_transform(ydata_samp)).explained_variance_ratio_)
        print("Shape of X before adding additional axes:",np.shape(X[:,:n_pca_comp]))
        # Put the PCA result into the orbital param df
        df_cut_orb['PCA_X'] = X[:,0]
        df_cut_orb['PCA_Y'] = X[:,1]
        df_cut_orb['PCA_least'] = X[:,-1]
        # Reduce the PCA result down to only the needed components
        X = X[:,:n_pca_comp]
        axes_labels = [r'PCA_X',r'PCA_Y']
    else:
        print("Axes not in PCA:",pca_list)
        # Put some place holder into the orbital param df
        df_cut_orb['PCA_X'] = np.empty(len(df_cut_orb))*np.nan
        df_cut_orb['PCA_Y'] = np.empty(len(df_cut_orb))*np.nan
        axes_labels = pca_list.copy()

    X_tp = X.T

    # Combine the result with the non-PCA axes and store the correponding uncertainties into a separate 
    # Do vstack with .T transpose twice!
    if action_add == True and diamond == True:
        X_tp = np.vstack((X_tp,df_cut_orb['diamond_x']))
        X_tp = np.vstack((X_tp,df_cut_orb['diamond_y']))
        axes_labels.append('Act_diamond')

    if action_add == True and diamond == False:
        X_tp = np.vstack((X_tp,df_cut_orb['JR']))
        X_tp = np.vstack((X_tp,df_cut_orb['Jphi']))
        X_tp = np.vstack((X_tp,df_cut_orb['Jz']))
        axes_labels.append('Act_3d')

    if metallicity_add == True:
        X_tp = np.vstack((X_tp,df_cut_vel['m_h_mean']))
        axes_labels.append('[Fe/H]')

    if velocity_add == True and cylindrical == False:
        X_tp = np.vstack((X_tp,df_cut_vel["U_g2"]))
        X_tp = np.vstack((X_tp,df_cut_vel["V_g2"]))
        X_tp = np.vstack((X_tp,df_cut_vel["W_g2"]))
        axes_labels.append('Vel_cart')

    if velocity_add == True and cylindrical == True:
        X_tp = np.vstack((X_tp,df_cut_vel["vr_g2"]))
        X_tp = np.vstack((X_tp,df_cut_vel["vphi_g2"]))
        X_tp = np.vstack((X_tp,df_cut_vel["vz_g2"]))
        axes_labels.append('Vel_cyl')

    if position_add == True:
        X_tp = np.vstack((X_tp,df_cut_vel["XGC"]))
        X_tp = np.vstack((X_tp,df_cut_vel["YGC"]))
        X_tp = np.vstack((X_tp,df_cut_vel["ZGC"]))
        axes_labels.append('Pos_cart')

    if energy_add == True:
        X_tp = np.vstack((X_tp,df_cut_orb["Etot"]))
        axes_labels.append('E_tot')

    if eccentricity_add == True:
        X_tp = np.vstack((X_tp,df_cut_orb["ecc"]))
        axes_labels.append('ecc')

    if Lz_add == True:
        X_tp = np.vstack((X_tp,df_cut_orb["Lz"]))
        axes_labels.append('Lz')

    if Lperp_add == True:
        X_tp = np.vstack((X_tp,df_cut_orb["Lperp"]))
        axes_labels.append('Lperp')


    # Put everything through Scaler again
    if scaler == 'Standard':
        X = StandardScaler().fit_transform(X_tp.T)
    elif scaler == 'Robust':
        X = RobustScaler().fit_transform(X_tp.T)
    elif scaler == None:
        X = X_tp.T
    
    if i == 0:
        print("Final clustering axes:",axes_labels)
        print("Shape of X after adding additional axes:",np.shape(X))
        print("Extra text is:",extratext)
    
    # Prepare arrays to store the results
    n_clusters_estimated = 60
    
    if denoise == 'PCA' and i == 0:
        print('counting PCA...')
        n_dim=2
    elif denoise != 'PCA' and i == 0:
        n_dim=0
        if metallicity == True:
            print('counting Metallicity...')
            n_dim += 1
        if action == True and diamond == True:
            print('counting action diamond...')
            n_dim += 2
        if action == True and diamond == False:
            print('counting 3d action...')
            n_dim += 3
        if velocity == True and cylindrical == False:
            print('counting cartesian velocity...')
            n_dim += 3
        if velocity == True and cylindrical == True:
            print('counting cylindrical velocity...')
            n_dim += 3
        if position == True:
            print('counting positions...')
            n_dim += 3
        if energy == True:
            print('counting energy...')
            n_dim += 1
        if eccentricity == True:
            print('counting eccentricity...')
            n_dim += 1
        if Lz == True:
            print('counting Lz...')
            n_dim += 1
        if Lperp == True:
            print('counting Lperp...')
            n_dim += 1

    if metallicity_add == True and i == 0:
        print('counting Metallicity...')
        n_dim += 1
    if action_add == True and diamond == True and i == 0:
        print('counting action diamond...')
        n_dim += 2
    if action_add == True and diamond == False and i == 0:
        print('counting 3d action...')
        n_dim += 3
    if velocity_add == True and i == 0:
        print('counting cartesian velocity...')
        n_dim += 3
    if position_add == True and i == 0:
        print('counting positions...')
        n_dim += 3
    if energy_add == True and i == 0:
        print('counting energy...')
        n_dim += 1
    if eccentricity_add == True and i == 0:
        print('counting eccentricity...')
        n_dim += 1
    if Lz_add == True and i == 0:
        print('counting Lz...')
        n_dim += 1
    if Lperp_add == True and i == 0:
        print('counting Lperp...')
        n_dim += 1


    cluster_means = np.zeros((n_clusters_estimated, n_dim))
    cluster_dispersions = np.zeros((n_clusters_estimated, n_dim))    
    cluster_nstars = np.zeros((n_clusters_estimated))
    if i == 0:
        print(n_dim)

    
    # Start clustering
    print("Start clustering")
    if algorithm == 'DBSCAN':
        # Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
    elif algorithm == 'OPTICS':
        # Apply OPTICS
        clust = OPTICS(min_samples=min_samples, max_eps = 2).fit(X)
        labels = clust.labels_ #[clust.ordering_]
    #         core_samples_mask = np.zeros_like(labels, dtype=bool)
    #         core_samples_mask[clust.core_sample_indices_] = True
    elif algorithm == 'HDBSCAN':
        # Apply HDBSCAN
        clust = hdbscan.HDBSCAN(min_samples=min_samples,
                                min_cluster_size=min_cluster_size,  
                                gen_min_span_tree=True, 
                                cluster_selection_method=cluster_selection_method).fit(X)
    #         clust = hdbscan.HDBSCAN().fit(X)
        labels = clust.labels_ #[clust.ordering_]
    elif algorithm == 'AGG_n' or algorithm == 'AGG_l':
        clust = AgglomerativeClustering(n_clusters=n_cluster,distance_threshold=linkage_threshold).fit(X)
        labels = clust.labels_
    elif algorithm == 'GMM':
        # Fix the random seed
        np.random.seed(0)
        # Find the best N_clust
        N = np.arange(3, n_clusters_estimated)
        models = [None for n in N]

        for i in range(len(N)):
            models[i] = GaussianMixture(N[i], covariance_type='full',reg_covar=1e-06).fit(X)

        AIC = [m.aic(X) for m in models]
        BIC = [m.bic(X) for m in models]

        # Auto determine the best GMM cluster number
        i_best = np.argmin(BIC)
        # Manual determine the best GMM cluster number
        i_best = 6
        print('best fit N components manually set!')
        gmm_best = models[i_best]
        print('best fit converged:', gmm_best.converged_)
        print('number of interations =', gmm_best.n_iter_)
        print('BIC: N components = %i' % N[i_best])

        plt.plot(N, AIC, 'r-', label='AIC')
        plt.plot(N, BIC, 'b--', label='BIC')
        plt.xlabel('number of components')
        plt.ylabel('information criterion')
        plt.legend(loc=2, frameon=False)
        plt.show()

        labels = gmm_best.predict(X)



    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))



    # ###########################################################################

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    print("unique_labels are", unique_labels)

    colors = [plt.cm.hsv(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
      
    
    # Make an empty list to save the indices
    mask_list = []
    col_list = []
    lab_list = []
    id_list = []

    # Make a separate source_id array for later extraction
    # Used to accomodate situations where we have additional cuts on top of the 
    # initial quality cut
    ind_add_cut = ind_cut[:]
    X_source_id = df_cut_vel.loc[ind_add_cut,'source_id']


    for k, col in zip(unique_labels, colors):
        print("k is", k)
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        if algorithm == 'DBSCAN':
            mask = class_member_mask & core_samples_mask
            anti_mask = class_member_mask & ~core_samples_mask
        elif algorithm == 'OPTICS' or algorithm == 'HDBSCAN' or algorithm == 'AGG_n' or algorithm == 'AGG_l' or algorithm == 'GMM':
            mask = class_member_mask
            anti_mask = class_member_mask

        
        xy = ydata_samp[mask]
        cluster_means[k] = np.mean(xy, axis = 0)
        cluster_dispersions[k] = np.std(xy, axis = 0)
        cluster_nstars[k] = len(X[class_member_mask])
        
        # Record the indices and colors
        mask_list.append(ind_add_cut[mask])
        id_list.append(X_source_id[mask].values)
        col_list.append(np.array(col*len(X[mask])).reshape((len(X[mask]),4)))
        lab_list.append(str(k))


    print("Shape of member stars", np.shape(mask_list))
    print("Exporting results...")
    np.save(data_dir + 'member_mask' + extratext + '_samp_' + str(i) + '.npy', mask_list)
    np.save(data_dir + 'member_gedr3id' + extratext + '_samp_' + str(i) + '.npy', id_list)
    np.save(data_dir + 'color' + extratext + '_samp_' + str(i) + '.npy', colors) # Note that the last color correcponds to noise and is not used
    np.save(data_dir + 'member_means' + extratext + '_samp_' + str(i) + '.npy', cluster_means)
    np.save(data_dir + 'member_disps' + extratext + '_samp_' + str(i) + '.npy', cluster_dispersions)
    np.save(data_dir + 'member_nstar' + extratext + '_samp_' + str(i) + '.npy', cluster_nstars)
    tree_df = clust.condensed_tree_.to_pandas()
    tree_df.to_pickle(data_dir + 'tree' + extratext + '_samp_' + str(i) + '.pkl',compression=None)

    # Decide if we want to make the plots
    if plotting_flag == False or i >= 20:
        continue
    

    # Generate the hierarchical tree if HDBSCAN
    if algorithm == 'HDBSCAN':
        f = plt.figure(figsize=[20,10])
        clust.condensed_tree_.plot(select_clusters=True, selection_palette=colors[:-1])
        f.savefig(plots_dir + 'clustering' + extratext + '_tree' + '_samp_' + str(i) + '.png')

    # Plot result (my summary)
    if n_noise_ == 0:
        plot_summary_v3(df_cut_vel,df_cut_orb,ind_list=mask_list,c_list=col_list,
                        title=extratext,legends=lab_list,list_mask=False,ind_add_cut=ind_add_cut,ds=False,noise=False,
                        filename=plots_dir+'summary'+extratext+ '_samp_' + str(i) +'_v3.png')
#         plot_summary_v3(df_cut_vel,df_cut_orb,ind_list=mask_list,c_list=col_list,
#                         title=extratext,legends=lab_list,list_mask=False,ind_add_cut=ind_add_cut,ds=False,noise=False,
#                         filename=plots_dir+'summary'+extratext+ '_samp_' + str(i) +'_v3_no_noise.png')
    else:
        plot_summary_v3(df_cut_vel,df_cut_orb,ind_list=mask_list,c_list=col_list,
                        title=extratext,legends=lab_list,list_mask=False,ind_add_cut=ind_add_cut,ds=False,
                        filename=plots_dir+'summary'+extratext+ '_samp_' + str(i) +'_v3.png')
#         plot_summary_v3(df_cut_vel,df_cut_orb,ind_list=mask_list[:-1],c_list=col_list[:-1],
#                         title=extratext,legends=lab_list[:-1],list_mask=False,ind_add_cut=ind_add_cut,ds=False,noise=False,
#                         filename=plots_dir+'summary'+extratext+ '_samp_' + str(i) +'_v3_no_noise.png')