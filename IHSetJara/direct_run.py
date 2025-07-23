# import fast_optimization as fo
# import pandas as pd
# import numpy as np
# import xarray as xr
# from scipy.stats import circmean
# from scipy.optimize import fsolve
# from IHSetJara import jara_njit
# from IHSetUtils import BreakingPropagation, ADEAN, hunt, Hs12Calc, depthOfClosure
# import json

# class Jara_run(object):
#     """
#     Jara_run
    
#     Configuration to calibrate and run the Jara et al. (2015) Shoreline Evolution Model.
    
#     This class reads input datasets, performs its calibration.
#     """

#     def __init__(self, path):

#         self.path = path
#         self.name = 'Jara et al. (2015)'
#         self.mode = 'standalone'
#         self.type = 'CS'
     
#         data = xr.open_dataset(path)
        
#         cfg = json.loads(data.attrs['run_Jara'])
#         self.cfg = cfg

#         self.switch_Yini = cfg['switch_Yini']
#         self.switch_brk = cfg['switch_brk']
#         if self.switch_brk == 1:
#             self.breakType = cfg['break_type']
#         self.doc_formula = cfg['doc_formula']
#         self.xc = cfg['xc']
#         self.Hberm = cfg['Hberm']
#         self.theta_max = cfg['theta_max']
#         self.D50 = cfg['D50']

#         if cfg['trs'] == 'Average':
#             self.hs = np.mean(data.hs.values, axis=1)
#             self.tp = np.mean(data.tp.values, axis=1)
#             self.dir = circmean(data.dir.values, high=360, low=0, axis=1)
#             self.time = pd.to_datetime(data.time.values)
#             self.Obs = data.average_obs.values
#             self.Obs = self.Obs[~data.mask_nan_average_obs]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_average_obs]
#             self.depth = np.mean(data.waves_depth.values)
#             self.bathy_angle = circmean(data.phi.values, high=360, low=0)
#         else:
#             self.hs = data.hs.values[:, cfg['trs']]
#             self.tp = data.tp.values[:, cfg['trs']]
#             self.dir = data.dir.values[:, cfg['trs']]
#             self.time = pd.to_datetime(data.time.values)
#             self.Obs = data.obs.values[:, cfg['trs']]
#             self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
#             self.depth = data.waves_depth.values[cfg['trs']]
#             self.bathy_angle = data.phi.values[cfg['trs']]

#         self.start_date = pd.to_datetime(cfg['start_date'])
#         self.end_date = pd.to_datetime(cfg['end_date'])

#         data.close()
        
#         self.hs12, self.tp12 = Hs12Calc(self.hs.reshape(-1, 1), self.tp.reshape(-1, 1))
#         self.hc = depthOfClosure(self.hs12, self.tp12, self.doc_formula)
#         self.theta_max = self.theta_max * np.pi / 180

#         if self.switch_brk == 0:
#             self.hb = self.hs
#             self.dirb = self.dir
#             self.depthb = self.hs/0.55
#         elif self.switch_brk == 1:
#             self.hb, self.dirb, self.depthb = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)

#         self.Ar = ADEAN(self.D50)
#         self.L = hunt(self.tp, self.depthb)

#         rhos = 2650
#         rho = 1025
#         g = 9.81  
#         ss = rhos/rho
#         self.gamma = 0.55

#         Ub_cr = (0.014*self.tp*((ss-1)**2)*(g**2)*(self.D50))**(1/3)
#         self.Hcr = (2**0.5/np.pi)*Ub_cr*self.tp*np.sinh((2*np.pi*self.depthb)/self.L)

#         self.split_data()

#         if self.switch_Yini == 1:
#             ii = np.argmin(np.abs(self.time_obs - self.time[0]))
#             self.Yini = self.Obs[ii]
#         self.Sm = np.mean(self.Obs)
#         self.xr_max = max(self.Obs)

#         xr_minimorum = self.xc - (self.hc / self.Ar) ** (3 / 2)
#         def equation(xr_min):
#             return 3/5 * self.Ar * (self.xc - xr_min) ** (5/3) + self.Hberm * (self.xc - xr_min) - (self.xc - self.xr_max) * (3/5 * self.hc + self.Hberm)
#         xr_min = fsolve(equation, xr_minimorum)

#         hb_maximorum = self.Ar * (self.xc - xr_min)**(2/3)
#         self.Vol = (self.xc - self.xr_max) * (3/5 * self.hc + self.Hberm)

#         def f(x):
#             return ((self.hc - x) / np.tan(self.theta_max) + (x**(3/2) - self.hc**(3/2)) / ((3/5 * (self.hc**(5/2) - x**(5/2)) + self.Hberm * (self.hc**(3/2) - x**(3/2))) / (self.Vol - (3/5 * x**(5/2) + self.Hberm * x**(3/2)) / self.Ar**(3/2))))

#         hb_max = fsolve(f, hb_maximorum)
#         self.hb_ = np.arange(0, hb_max + 0.01, 0.01)
#         if  self.hb_[-1] < hb_max:
#             self.hb_ = np.append( self.hb_, hb_max)
        
#         Ee_ = (self.gamma**2 *  self.hb_**2) / 4.004**2

#         self.xre_ = self.xc - ((self.hb_ / self.Ar) ** (3 / 2)) + (self.hb_**(3/2) - self.hc**(3/2)) / (
#                 (3/5 * (self.hc**(5/2) - self.hb_**(5/2)) + self.Hberm * (self.hc**(3/2) - self.hb_**(3/2))) /
#                 (self.Vol - (3/5 * self.hb_**(5/2) + self.Hberm * self.hb_**(3/2)) / self.Ar**(3/2)))

#         AA = np.array([[self.xre_[0]**2, self.xre_[0], 1], [self.xre_[-1]**2, self.xre_[-1], 1], [2*self.xre_[0], 1, 0]])
#         BB = np.array([Ee_[0], Ee_[-1], 0])
#         self.pol = np.linalg.lstsq(AA, BB, rcond=None)[0]
        
#         mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        
#         self.idx_obs = mkIdx(self.time_obs)

#         # Now we calculate the dt from the time variable
#         mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
#         self.dt = mkDT(np.arange(0, len(self.time)-1))

#         if self.switch_Yini == 0:
#             def run_model(par):
#                 ca = par[0]
#                 ce = par[1]
#                 Yini = par[2]
#                 Ymd = jara_njit(self.hb,
#                               self.Hcr,
#                               Yini,
#                               self.dt,
#                               self.gamma,
#                               self.xc,
#                               self.hc,
#                               self.Hberm,
#                               self.Ar,
#                               self.hb_,
#                               self.xre_,
#                               self.pol,
#                               self.Vol,
#                               ca,
#                               ce)
#                 return Ymd
        
#             self.run_model = run_model
#         else:
#             def run_model(par):
#                 ca = par[0]
#                 ce = par[1]
#                 Ymd = jara_njit(self.hb,
#                               self.Hcr,
#                               self.Yini,
#                               self.dt,
#                               self.gamma,
#                               self.xc,
#                               self.hc,
#                               self.Hberm,
#                               self.Ar,
#                               self.hb_,
#                               self.xre_,
#                               self.pol,
#                               self.Vol,
#                               ca,
#                               ce)
#                 return Ymd
        
#             self.run_model = run_model
    
#     def run(self, par):
#         self.full_run = self.run_model(par)
#         if self.switch_Yini == 1:
#             self.par_names = [r'C+', r'C-']
#             self.par_values = par
#         elif self.switch_Yini == 0:
#             self.par_names = [r'C+', r'C-', r'Y_{i}']
#             self.par_values = par
        
#         # self.calculate_metrics()

#     def calculate_metrics(self):
#         self.metrics_names = fo.backtot()[0]
#         self.indexes = fo.multi_obj_indexes(self.metrics_names)
#         self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)

#     def split_data(self):
#         """
#         Split the data into calibration and validation datasets.
#         """
#         ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
#         self.hb = self.hb[ii]
#         self.Hcr = self.Hcr[ii]
#         self.time = self.time[ii]

#         ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
#         self.Obs = self.Obs[ii]
#         self.time_obs = self.time_obs[ii]



import numpy as np
from scipy.optimize import fsolve
from IHSetJara import jara
from IHSetUtils import ADEAN, hunt, Hs12Calc, depthOfClosure
from IHSetUtils.CoastlineModel import CoastlineModel

class Jara_run(CoastlineModel):
    """
    Jara_run
    
    Configuration to calibrate and run the Jara et al. (2015) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Jara et al. (2015)',
            mode='standalone',
            model_type='CS',
            model_key='run_Jara'
        )

        self.setup_forcing()

    def setup_forcing(self):
        
        self.switch_Yini = self.cfg['switch_Yini']
        self.doc_formula = self.cfg['doc_formula']
        self.xc = self.cfg['xc']
        self.Hberm = self.cfg['Hberm']
        self.theta_max = self.cfg['theta_max']
        self.D50 = self.cfg['D50']
        
        self.hs12, self.tp12 = Hs12Calc(self.hs.reshape(-1, 1), self.tp.reshape(-1, 1))
        self.hc = depthOfClosure(self.hs12, self.tp12, self.doc_formula)
        self.theta_max = self.theta_max * np.pi / 180

        self.Ar = ADEAN(self.D50)

        self.tp[self.tp < 0.1] = 0.1
        self.depthb[self.depthb < 0.1] = 0.1
        self.hb[self.hb < 0.01] = 0.01

        self.L = hunt(self.tp, self.depthb)
        
        rhos = 2650
        rho = 1025
        g = 9.81  
        ss = rhos/rho
        self.gamma = 0.55

        Ub_cr = (0.014*self.tp*((ss-1)**2)*(g**2)*(self.D50))**(1/3)
        self.Hcr = (2**0.5/np.pi)*Ub_cr*self.tp*np.sinh((2*np.pi*self.depthb)/self.L)
        self.Hcr_s  = (2**0.5/np.pi)*Ub_cr*self.tp*np.sinh((2*np.pi*self.depthb_s)/self.L)

        if self.switch_Yini == 1:
            self.Yini = self.Obs[0]
        self.Sm = np.mean(self.Obs)
        self.xr_max = max(self.Obs)

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
        
    def init_par(self, population_size: int):

        if self.switch_Yini == 0:
            lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1])])
            uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1])])
        else:
            lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1]), 0.75*np.min(self.Obs)])
            uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1]), 1.25*np.max(self.Obs)])
        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers
    
    def run_model(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 1:
            ca = par[0]
            ce = par[1]
            Ymd, _ = jara(self.hb,
                            self.Hcr,
                            self.Yini,
                            self.dt,
                            self.gamma,
                            self.xc,
                            self.hc,
                            self.Hberm,
                            self.Ar,
                            self.xre_,
                            self.pol,
                            self.Vol,
                            ca,
                            ce)
        elif self.switch_Yini == 0:
            ca = par[0]
            ce = par[1]
            Yini = par[2]
            Ymd, _ = jara(self.hb,
                            self.Hcr,
                            Yini,
                            self.dt,
                            self.gamma,
                            self.xc,
                            self.hc,
                            self.Hberm,
                            self.Ar,
                            self.xre_,
                            self.pol,
                            self.Vol,
                            ca,
                            ce)
        return Ymd

    def _set_parameter_names(self):
        if self.switch_Yini == 1:
            self.par_names = [r'C+', r'C-']
        elif self.switch_Yini == 0:
            self.par_names = [r'C+', r'C-', r'Y_i']




