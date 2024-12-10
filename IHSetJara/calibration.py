import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from scipy.optimize import fsolve
from IHSetJara import jara
from IHSetCalibration import objective_functions
from IHSetUtils import BreakingPropagation, ADEAN, hunt, Hs12Calc, depthOfClosure

class cal_Jara(object):
    """
    cal_Jara
    
    Configuration to calibfalse,and run the Jara et al. (2015) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')


        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = cfg['dt'].values
        self.switch_Yini = cfg['switch_Yini'].values

        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)

        self.Hs = wav['Hs'].values
        self.Tp = wav['Tp'].values
        self.Dir = wav['Dir'].values
        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)

        self.Obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))
        
        self.depth = cfg['depth'].values
        self.angleBathy = cfg['bathy_angle'].values
        H12,T12 = Hs12Calc(self.Hs.reshape(-1, 1),self.Tp.reshape(-1, 1))
        DoC = depthOfClosure(H12,T12)
        self.hc = DoC[0][0]
        the_max = cfg['theta_max'].values
        self.theta_max = the_max * np.pi / 180
        self.xc = cfg['xc'].values
        self.B = cfg['Hberm'].values
        self.D50 = cfg['D50'].values

        breakType = "spectral"
        self.Hb, self.theb, self.depthb = BreakingPropagation(self.Hs, self.Tp, self.Dir,
                                                              np.repeat(self.depth, (len(self.Hs))), np.repeat(self.angleBathy, (len(self.Hs))),  breakType)
        self.Ar = ADEAN(self.D50)        
        self.L = hunt(self.Tp, self.depthb)
           
        rhos = 2650
        rho = 1025
        g = 9.81  
        
        ss = rhos/rho
        self.gamma = 0.55
        Ub_cr = (0.014*self.Tp*((ss-1)**2)*(g**2)*(self.D50))**(1/3)
        self.Hcr = (2**0.5/np.pi)*Ub_cr*self.Tp*np.sinh((2*np.pi*self.depthb)/self.L)
        
        self.split_data()
        
        if self.switch_Yini == 0:
            self.S0 = self.Obs_splited[0]
        self.Sm = np.mean(self.Obs_splited)
        self.xr_max = max(self.Obs_splited)
             
        xr_minimorum = self.xc - (self.hc / self.Ar) ** (3 / 2)
        def equation(xr_min):
            return 3/5 * self.Ar * (self.xc - xr_min) ** (5/3) + self.B * (self.xc - xr_min) - (self.xc - self.xr_max) * (3/5 * self.hc + self.B)
        xr_min = fsolve(equation, xr_minimorum)
    
        hb_maximorum = self.Ar * (self.xc - xr_min)**(2/3)
        self.Vol = (self.xc - self.xr_max) * (3/5 * self.hc + self.B)
        def f(x):
            return ((self.hc - x) / np.tan(self.theta_max) + (x**(3/2) - self.hc**(3/2)) /
                    ((3/5 * (self.hc**(5/2) - x**(5/2)) + self.B * (self.hc**(3/2) - x**(3/2))) /
                     (self.Vol - (3/5 * x**(5/2) + self.B * x**(3/2)) / self.Ar**(3/2))))        
        hb_max = fsolve(f, hb_maximorum)
        self.hb_ = np.arange(0, hb_max + 0.01, 0.01)
        if  self.hb_[-1] < hb_max:
             self.hb_ = np.append( self.hb_, hb_max)

        Ee_ = (self.gamma**2 *  self.hb_**2) / 4.004**2
        self.xre_ = self.xc - ((self.hb_ / self.Ar) ** (3 / 2)) + (self.hb_**(3/2) - self.hc**(3/2)) / (
                (3/5 * (self.hc**(5/2) - self.hb_**(5/2)) + self.B * (self.hc**(3/2) - self.hb_**(3/2))) /
                (self.Vol - (3/5 * self.hb_**(5/2) + self.B * self.hb_**(3/2)) / self.Ar**(3/2)))
        AA = np.array([[self.xre_[0]**2, self.xre_[0], 1], [self.xre_[-1]**2, self.xre_[-1], 1], [2*self.xre_[0], 1, 0]])
        BB = np.array([Ee_[0], Ee_[-1], 0])
        self.pol = np.linalg.lstsq(AA, BB, rcond=None)[0]
                
        cfg.close()
        wav.close()
        ens.close()
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        if self.switch_Yini == 0:
            def model_simulation(par):
                Ca = par['Ca']
                Ce = par['Ce']
                Ymd = jara.jara(self.Hb_splited,
                                    self.Hcr_splited,
                                    self.S0,
                                    self.dt,
                                    self.gamma,
                                    self.xc,
                                    self.hc,
                                    self.B,
                                    self.Ar,
                                    self.hb_,
                                    self.xre_,
                                    self.pol,
                                    self.Vol,
                                    Ca,
                                    Ce)
                
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('Ca', -1e-1, -1e-10),
                Uniform('Ce', -1e-1, -1e-10)
            ]
            self.model_sim = model_simulation

        elif self.switch_Yini == 1:
            def model_simulation(par):
                Ca = par['Ca']
                Ce = par['Ce']
                Yini = par['Yini']
                Ymd = jara.jara(self.Hb_splited,
                                    self.Hcr_splited,
                                    Yini,
                                    self.dt,
                                    self.gamma,
                                    self.xc,
                                    self.hc,
                                    self.B,
                                    self.Ar,
                                    self.hb_,
                                    self.xre_,
                                    self.pol,
                                    self.Vol,
                                    Ca,
                                    Ce)
                
                return Ymd[self.idx_obs_splited]
                
            self.params = [
                Uniform('Ca', -1e-1, -1e-10),
                Uniform('Ce', -1e-1, -1e-10),
                Uniform('Yini', np.min(self.Obs), np.max(self.Obs))
            ]
            self.model_sim = model_simulation


    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.Hb_splited = self.Hb[idx]
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
