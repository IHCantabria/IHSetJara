import fast_optimization as fo
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import circmean
from spotpy.parameter import Uniform
from scipy.optimize import fsolve
from IHSetJara import jara
from IHSetCalibration import objective_functions
from IHSetUtils import BreakingPropagation, ADEAN, hunt, Hs12Calc, depthOfClosure
import json

class cal_Jara_2(object):
    """
    cal_Jara_2
    
    Configuration to calibfalse,and run the Jara et al. (2015) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['Jara'])

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.switch_Yini = cfg['switch_Yini']
        self.switch_brk = cfg['switch_brk']
        if self.switch_brk == 1:
            self.bathy_angle = cfg['bathy_angle']
            self.breakType = cfg['break_type']
            self.depth = cfg['depth']
        self.doc_formula = cfg['doc_formula']
        self.xc = cfg['xc']
        self.Hberm = cfg['Hberm']
        self.theta_max = cfg['theta_max']
        self.D50 = cfg['D50']
        self.lb = cfg['lb']
        self.ub = cfg['ub']

        self.calibr_cfg = fo.config_cal(cfg)

        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.tp = np.mean(data.tp.values, axis=1)
            self.dir = circmean(data.dir.values, high=360, low=0, axis=1)
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        data.close()

        self.hs12, self.tp12 = Hs12Calc(self.hs.reshape(-1, 1), self.tp.reshape(-1, 1))
        self.hc = depthOfClosure(self.hs12, self.tp12, self.doc_formula)
        self.theta_max = self.theta_max * np.pi / 180

        if self.switch_brk == 0:
            self.hb = self.hs
            self.dirb = self.dir
            self.depthb = self.hs/0.55
        elif self.switch_brk == 1:
            self.hb, self.dirb, self.depthb = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)

        self.Ar = ADEAN(self.D50)
        self.L = hunt(self.tp, self.depthb)

        rhos = 2650
        rho = 1025
        g = 9.81  
        ss = rhos/rho
        self.gamma = 0.55

        Ub_cr = (0.014*self.tp*((ss-1)**2)*(g**2)*(self.D50))**(1/3)
        self.Hcr = (2**0.5/np.pi)*Ub_cr*self.tp*np.sinh((2*np.pi*self.depthb)/self.L)

        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]
        self.Sm = np.mean(self.Obs_splited)
        self.xr_max = max(self.Obs_splited)

        xr_minimorum = self.xc - (self.hc / self.Ar) ** (3 / 2)
        def equation(xr_min):
            return 3/5 * self.Ar * (self.xc - xr_min) ** (5/3) + self.Hberm * (self.xc - xr_min) - (self.xc - self.xr_max) * (3/5 * self.hc + self.Hberm)
        xr_min = fsolve(equation, xr_minimorum)

        hb_maximorum = self.Ar * (self.xc - xr_min)**(2/3)
        self.Vol = (self.xc - self.xr_max) * (3/5 * self.hc + self.Hberm)

        def f(x):
            return ((self.hc - x) / np.tan(self.theta_max) + (x**(3/2) - self.hc**(3/2)) / ((3/5 * (self.hc**(5/2) - x**(5/2)) + self.Hberm * (self.hc**(3/2) - x**(3/2))) / (self.Vol - (3/5 * x**(5/2) + self.Hberm * x**(3/2)) / self.Ar**(3/2))))

        hb_max = fsolve(f, hb_maximorum)
        self.hb_ = np.arange(0, hb_max + 0.01, 0.01)
        if  self.hb_[-1] < hb_max:
            self.hb_ = np.append( self.hb_, hb_max)
        
        Ee_ = (self.gamma**2 *  self.hb_**2) / 4.004**2

        self.xre_ = self.xc - ((self.hb_ / self.Ar) ** (3 / 2)) + (self.hb_**(3/2) - self.hc**(3/2)) / (
                (3/5 * (self.hc**(5/2) - self.hb_**(5/2)) + self.Hberm * (self.hc**(3/2) - self.hb_**(3/2))) /
                (self.Vol - (3/5 * self.hb_**(5/2) + self.Hberm * self.hb_**(3/2)) / self.Ar**(3/2)))

        AA = np.array([[self.xre_[0]**2, self.xre_[0], 1], [self.xre_[-1]**2, self.xre_[-1], 1], [2*self.xre_[0], 1, 0]])
        BB = np.array([Ee_[0], Ee_[-1], 0])
        self.pol = np.linalg.lstsq(AA, BB, rcond=None)[0]
        
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))


        if self.switch_Yini == 0:
            # @jit
            def model_simulation(par):
                ca = -np.exp(par[0])
                ce = -np.exp(par[1])
                
                Ymd = jara(self.hb_splited,
                              self.Hcr_splited,
                              self.Yini,
                              self.dt_splited,
                              self.gamma,
                              self.xc,
                              self.hc,
                              self.Hberm,
                              self.Ar,
                              self.hb_,
                              self.xre_,
                              self.pol,
                              self.Vol,
                              ca,
                              ce)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                ca = -np.exp(par[0])
                ce = -np.exp(par[1])
                Ymd = jara(self.hb,
                              self.Hcr,
                              self.Yini,
                              self.dt,
                              self.gamma,
                              self.xc,
                              self.hc,
                              self.Hberm,
                              self.Ar,
                              self.hb_,
                              self.xre_,
                              self.pol,
                              self.Vol,
                              ca,
                              ce)
                return Ymd

            self.run_model = run_model

            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1])])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1])])
                population = np.zeros((population_size, 2))
                for i in range(2):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            def model_simulation(par):
                ca = -np.exp(par[0])
                ce = -np.exp(par[1])
                Yini = par[2]
                Ymd = jara(self.hb_splited,
                              self.Hcr_splited,
                              Yini,
                              self.dt_splited,
                              self.gamma,
                              self.xc,
                              self.hc,
                              self.Hberm,
                              self.Ar,
                              self.hb_,
                              self.xre_,
                              self.pol,
                              self.Vol,
                              ca,
                              ce)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def model_simulation(par):
                ca = -np.exp(par[0])
                ce = -np.exp(par[1])
                Yini = par[2]
                Ymd = jara(self.hb,
                              self.Hcr,
                              Yini,
                              self.dt,
                              self.gamma,
                              self.xc,
                              self.hc,
                              self.Hberm,
                              self.Ar,
                              self.hb_,
                              self.xre_,
                              self.pol,
                              self.Vol,
                              ca,
                              ce)
                return Ymd

            self.run_model = run_model

            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1]), 0.75*np.min(self.Obs)])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1]), 1.25*np.max(self.Obs)])
                population = np.zeros((population_size, 3))
                for i in range(3):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """

        ii = np.where(self.time>=self.start_date)[0][0]
        self.hb = self.hb[ii:]
        self.Hcr = self.Hcr[ii:]
        self.time = self.time[ii:]
        
        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.hb_splited = self.hb[idx]
        self.Hcr_splited = self.Hcr[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)
            