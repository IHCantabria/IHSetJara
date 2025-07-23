import numpy as np
from scipy.optimize import fsolve
from IHSetJara import jara
from IHSetUtils import ADEAN, hunt, Hs12Calc, depthOfClosure
from IHSetUtils.CoastlineModel import CoastlineModel

class cal_Jara_2(CoastlineModel):
    """
    cal_Jara_2
    
    Configuration to calibfalse,and run the Jara et al. (2015) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Jara et al. (2015)',
            mode='calibration',
            model_type='CS',
            model_key='Jara'
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
    
    def model_sim(self, par: np.ndarray) -> np.ndarray:
        
        if self.switch_Yini == 0:
            ca = -np.exp(par[0])
            ce = -np.exp(par[1])
            Ymd, _ = jara(self.hb_s,
                            self.Hcr_s,
                            self.Yini,
                            self.dt_s,
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
            
        elif self.switch_Yini == 1:
            ca = -np.exp(par[0])
            ce = -np.exp(par[1])
            Yini = par[2]
            Ymd, _ = jara(self.hb_s,
                            self.Hcr_s,
                            Yini,
                            self.dt_s,
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
        return Ymd[self.idx_obs_splited]
    
    def run_model(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 0:
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
        elif self.switch_Yini == 1:
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
        if self.switch_Yini == 0:
            self.par_names = [r'C+', r'C-']
        elif self.switch_Yini == 1:
            self.par_names = [r'C+', r'C-', r'Y_i']
        for idx in [0, 1]:
            self.par_values[idx] = -np.exp(self.par_values[idx])
