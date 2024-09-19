from IHSetJara import calibration, jara
from IHSetCalibration import setup_spotpy, mielke_skill_score
import xarray as xr
import os
import matplotlib.pyplot as plt
import spotpy as spt
import numpy as np

# Avaliable methods: NSGAII, SCE-UA

# config = xr.Dataset(coords={'dt': 3,                # [hours]
#                             'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
#                             'depth': 10,            # Water depth [m]
#                             'D50': 0.427e-3,        # Median grain size [m]
#                             'bathy_angle': 36,      # Bathymetry mean orientation [deg N]
#                             'xc' : 255,             # Cross-shore distance from shore to closure depth [m]
#                             'hc' : 6,               # Depth of closure [m]
#                             'B' : 2,                # Berm height [m]

#                             'Ysi': 1975,            # Initial year for calibration
#                             'Msi': 1,               # Initial month for calibration
#                             'Dsi': 1,               # Initial day for calibration
#                             'Ysf': 2013,            # Final year for calibration
#                             'Msf': 1,               # Final month for calibration
#                             'Dsf': 1,               # Final day for calibration
#                             'cal_alg': 'NSGAII',    # Avaliable methods: NSGAII
#                             'metrics': 'mss_nsse',  # Metrics to be minimized (mss_rmse, mss_rho, mss_rmse_rho)
#                             'n_pop': 50,            # Number of individuals in the population
#                             'n_obj': 2,             # Number of objectives to be minimized 
#                             'generations': 500,    # Number of generations for the calibration algorithm
#                             })              

config = xr.Dataset(coords={'dt': 3,                # [hours]
                            'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
                            'depth': 10,            # Water depth [m]
                            'D50': 0.427e-3,        # Median grain size [m]
                            'bathy_angle': 36,      # Bathymetry mean orientation [deg N]
                            'xc' : 255,             # Cross-shore distance from shore to closure depth [m]
                            'hc' : 6,               # Depth of closure [m]
                            'B' : 2,                # Berm height [m]
                                                    
                            'Ysi': 1975,            # Initial year for calibration
                            'Msi': 1,               # Initial month for calibration
                            'Dsi': 1,               # Initial day for calibration
                            'Ysf': 2013,            # Final year for calibration
                            'Msf': 1,               # Final month for calibration
                            'Dsf': 1,               # Final day for calibration
                            'cal_alg': 'sceua',     # Avaliable methods: sceua
                            'metrics': 'rmse',      # Metrics to be minimized (mss, RP, rmse, nsse)
                            'repetitions': 50000    # Number of repetitions for the calibration algorithm
                            })

wrkDir = os.getcwd()
config.to_netcdf(wrkDir+'/data/config.nc', engine='netcdf4')

model = calibration.cal_Jara(wrkDir+'/data/')

setup = setup_spotpy(model)

results = setup.setup()

bestindex, bestobjf = spt.analyser.get_minlikeindex(results)
best_model_run = results[bestindex]
fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
best_simulation = list(best_model_run[fields])

Ca = best_model_run['parCa']
Ce = best_model_run['parCe']
# S0 = best_model_run['parYini']

full_run = jara.jara(model.Hb, model.Hcr, model.Obs[0], model.dt, model.gamma, model.xc, model.hc, model.B, model.Ar, model.hb_, model.xre_, model.pol, model.Vol, Ca, Ce)
# full_run = jara.jara(model.Hb, model.Hcr, S0, model.dt, model.gamma, model.xc, model.hc, model.B, model.Ar, model.hb_, model.xre_, model.pol, model.Vol, Ca, Ce)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}

ylim_lower = np.floor(np.min([np.nanmin(model.Obs), np.nanmin(full_run)]) / 2) * 2
ylim_upper = np.ceil(np.max([np.nanmax(model.Obs), np.nanmax(full_run)]) / 2) * 2

fig, ax = plt.subplots(2 , 1, figsize=(10, 2), dpi=300, linewidth=5, edgecolor="#04253a", gridspec_kw={'height_ratios': [3.5, 1.5]})
# ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
ax[0].scatter(model.time_obs, model.Obs,s = 1, c = 'grey', label = 'Observed data')
ax[0].plot(model.time, full_run, color='red',linestyle='solid', label= 'Jara et al. (2015)')
ax[0].fill([model.start_date, model.end_date, model.end_date, model.start_date], [ylim_lower, ylim_lower, ylim_upper, ylim_upper], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
ax[0].set_ylim([ylim_lower,ylim_upper])
ax[0].set_xlim([model.time[0], model.time[-1]])
ax[0].set_ylabel('S [m]', fontdict=font)
ax[0].legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.20))
ax[0].grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)

ax[1].plot(model.time, model.Hb/np.max(model.Hb),color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
ax[1].set_ylim([0,1])
ax[1].set_xlim([model.time[0], model.time[-1]])
ax[1].set_yticks([0, 1], ['0', r'$H_{b max}$'])
plt.subplots_adjust(hspace=0.3)
fig.savefig('./results/Jara_Best_modelrun_'+str(config.cal_alg.values)+'.png',dpi=300)

# Calibration:
rmse = spt.objectivefunctions.rmse(model.observations, best_simulation)
nsse = spt.objectivefunctions.nashsutcliffe(model.observations, best_simulation)
mss = mielke_skill_score(model.observations, best_simulation)
rp = spt.objectivefunctions.rsquared(model.observations, best_simulation)
bias = spt.objectivefunctions.bias(model.observations, best_simulation)

# Validation:
run_cut = full_run[model.idx_validation]
rmse_v = spt.objectivefunctions.rmse(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
nsse_v = spt.objectivefunctions.nashsutcliffe(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
mss_v = mielke_skill_score(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
rp_v = spt.objectivefunctions.rsquared(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
bias_v = spt.objectivefunctions.bias(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])

print('Metrics                       | Calibration  | Validation|')
print('RMSE [m]                      | %-5.2f        | %-5.2f     |' % (rmse, rmse_v))
print('Nash-Sutcliffe coefficient [-]| %-5.2f        | %-5.2f     |' % (nsse, nsse_v))
print('Mielke Skill Score [-]        | %-5.2f        | %-5.2f     |' % (mss, mss_v))
print('R2 [-]                        | %-5.2f        | %-5.2f     |' % (rp, rp_v))
print('Bias [m]                      | %-5.2f        | %-5.2f     |' % (bias, bias_v))

config.close()