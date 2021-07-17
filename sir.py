import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)
import math
import numpy as np

def plot_sir(x, S, I, R, x0, x1, y0, y1, dis_x, dis_y, title):
    grafico, ax = plt.subplots(figsize = (12, 12));

    plt.xlabel("Days", fontsize = 16);
    plt.ylabel("People", fontsize = 16);

    plt.title(title, fontsize = 25)

    plt.tick_params(axis='both', which='major', labelsize = 10)
    plt.tick_params(axis='both', which='minor', labelsize = 10)

    ax.xaxis.set_major_locator( MultipleLocator(dis_x) )
    ax.yaxis.set_major_locator( MultipleLocator(dis_y) )

    plt.xlim([x0, x1]);
    plt.ylim([y0, y1]);
    plt.plot(x, S, color = "orange")
    plt.plot(x, I, color = "red")
    plt.plot(x, R, color = "green")
    
def sir(t, s, i, r, alpha, beta, N):
    ds = - alpha * s * i / N
    di = alpha * s * i / N - beta * i
    dr = beta * i
    return ds, di, dr
    
# Runge-Kutta 4 numerical integration algorithm
def rk4_sir(f, t, h, y, alpha, beta, N):
  s, i, r = y
  k1s, k1i, k1r = [ h * x for x in f(t, s, i, r, alpha, beta, N) ]
  k2s, k2i, k2r = [ h * x for x in f(t + h / 2, s + k1s / 2, i + k1i / 2, r + k1r / 2, alpha, beta, N) ]
  k3s, k3i, k3r = [ h * x for x in f(t + h / 2, s + k2s / 2, i + k2i / 2, r + k2r / 2, alpha, beta, N) ]
  k4s, k4i, k4r = [ h * x for x in f(t + h, s + k3s, i + k3i, r + k3r, alpha, beta, N) ]
  s = s + (k1s + 2 * k2s + 2 * k3s + k4s) / 6
  i = i + (k1i + 2 * k2i + 2 * k3i + k4i) / 6
  r = r + (k1r + 2 * k2r + 2 * k3r + k4r) / 6
  return s, i, r
  
alpha = 0.27    # Infection rate
beta = 0.043    # Recovery rate
N = 15000       # Population

i = 0.03 * N
s = N - i
r = 0
y = s, i, r

t_max = 150
t = np.linspace(0, t_max, t_max)
dt = 0.1

discretization = 10
ts = [ t / discretization  for t in range(0, t_max * discretization) ]
S = []
I = []
R = []

for t in ts:
  s, i, r = y
  S.append(s)
  I.append(i)
  R.append(r)
  y = rk4_sir(sir, t, dt, y, alpha, beta, N)
  
plot_sir(ts, S, I, R, 0, t_max, 0, N, 10, 1000, "Epidemic Evolution")

""" 
Version for scipy:

import scipy.integrate
from scipy.integrate import odeint

def sir(y, t):
    s, i, r = y
    alpha = 0.27 
    beta = 0.043 
    N = 10000
    ds = - alpha * s * i / N
    di = alpha * s * i / N - beta * i
    dr = beta * i
    return ds, di, dr

N = 10000
i = 0.03 * N 
r = 0
s = N - i - r
y = s, i, r
t_max = 200
t = np.linspace(0, t_max, t_max)

solution = odeint(sir_aux, y, t)
S, I, R = solution.T
plt.plot(t, S)
"""
