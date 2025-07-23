import numpy as np
from numba import njit
import math

@njit(fastmath=True, cache=True)
def P_evolutivo_parabolico_numba(a, b, xro, xre, C, ddt):
    # Parámetros precomputados a y b/a
    if xro == xre:
        return xro
    b_div_a = b / a
    exponent = (xro + xre + b_div_a) / (xro - xre) * math.exp(C * a * (2 * xre + b_div_a) * ddt)

    
    return (b_div_a + xre * (exponent + 1.0)) / (exponent - 1.0)

@njit(fastmath=True, cache=True)
def jara(Hb, Hcr, xrini, dt, gamma, xc, hc, B, Ar, xre_tpl, pol, Vol, Ca, Ce):
    n = Hb.shape[0]
    # Convertir dt a segundos (vector)
    dt_sec = dt * 3600.0
    
    # Preasignaciones
    xr = np.empty(n)
    xr[0] = xrini
    xre_pot = np.empty(n - 1)
    
    # Constantes
    a = pol[0]
    b = pol[1]
    xre_max = np.max(xre_tpl)
    xre_min = np.min(xre_tpl)
    
    for i in range(1, n):
        xro = xr[i - 1]
        hbe = Hb[i - 1] / gamma

        # Cálculo de xre potencial
        num = (hbe ** 1.5 - hc ** 1.5)
        den = (0.6 * (hc ** 2.5 - hbe ** 2.5) + B * (hc ** 1.5 - hbe ** 1.5))
        term = Vol - (0.6 * hbe ** 2.5 + B * hbe ** 1.5) / (Ar ** 1.5)
        xre_val = xc - (hbe / Ar) ** 1.5 + num / (den / term)

        # Decisión de acreción/erosión
        if Hb[i - 1] == 0.0 or Hb[i - 1] <= Hcr[i - 1]:
            xr_i = xro
        else:
            if xre_val > xro:
                xr_i = P_evolutivo_parabolico_numba(a, b, xro, xre_val, Ca, dt_sec[i - 1])
            else:
                xr_i = P_evolutivo_parabolico_numba(a, b, xro, xre_val, Ce, dt_sec[i - 1])

        # Clamping
        if xr_i > xre_max:
            xr_i = xre_max
        elif xr_i < xre_min:
            xr_i = xre_min

        xr[i] = xr_i
        xre_pot[i - 1] = xre_val

    return xr, xre_pot


def jara_njit(Hb_, Hcr_, xrini, dt, gamma, xc, hc, B, Ar, hbe_, xre_, pol, Vol, Ca, Ce):
    """
    Jara et al. 2015 model
    """
    
    def P_evolutivo_parabolico(pol, xro, xre, C, ddt):
        a = pol[0]
        b = pol[1]
        exponent = (xro + xre + b / a) / (xro - xre) * np.exp(C * a * (2 * xre + b / a) * ddt)
        xr = (b / a + xre * (exponent + 1)) / (exponent - 1)
        return xr
    
    dt = dt * 3600
    xr_ = np.empty(len(Hb_)-1)
    xr_[:]=0.0  # Initialize to zeros
    xre_potencial = np.empty(len(Hb_)-1)
    xre_potencial[:]=0.0  # Initialize to zeros

    for i in range(len(Hb_)-1): 
        if i == 0:
            xro = xrini
        if i >= 1:
            xro = xr_[i - 1]

        if xro == np.max(xre_):
            hbo = np.min(hbe_)
        elif xro == np.min(xre_):
            hbo = np.min(hbe_)
        else:
            hbo = np.interp(xro, xre_, hbe_)

        hbe = Hb_[i] / gamma
        Ee = (Hb_[i] / 4.004) ** 2

        # Obtengo la posición de equilibrio de la línea de costa
        xre = xc - (hbe / Ar) ** (3/2) + (hbe ** (3/2) - hc ** (3/2)) / (
            (3/5 * (hc ** (5/2) - hbe ** (5/2)) + B * (hc ** (3/2) - hbe ** (3/2))) /
            (Vol - (3/5 * hbe ** (5/2) + B * hbe ** (3/2)) / Ar ** (3/2)))

        if Hb_[i] == 0 or Hb_[i] <= Hcr_[i]:
            xr_[i] = xro
        elif xre > xro:  # Si se va a producir acreción
            xr_[i] = P_evolutivo_parabolico(pol, xro, xre, Ca, dt[i])
        else:  # Si se va a producir erosión
            xr_[i] = P_evolutivo_parabolico(pol, xro, xre, Ce, dt[i])
    
        # Para evitar que nos salgamos de los rangos de xrmin y xrmax
        if xr_[i] > np.max(xre_):
            xr_[i] = np.max(xre_)
            # idmax[i] = True
        elif xr_[i] < np.min(xre_):
            xr_[i] = np.min(xre_)
            # idmin[i] = True

        # Se guarda el xre de equilibrio de cada estado de mar
        xre_potencial[i] = xre
        
    # Para asignar a cada e.m. el valor del modelo en el instante inicial
    # xr_ = np.concatenate(([xrini], xr_[1:]))
    S = np.empty(len(xr_) + 1)
    S[0] = xrini
    S[1:] = xr_
    
    return S