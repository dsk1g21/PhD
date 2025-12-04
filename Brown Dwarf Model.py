import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# constants etc
k = 1.38e-23 # m2kgs-2K-1
e = 1.6e-19 # C to keep B in Teslas
au = 1.67e-27 # kg
T_eff = 800 # K based on the rodriguez paper
logg = 5 # cms-1
P0 = 1.2e-4 # bar
g = 10 ** logg * 1e-2 # to adjust and get ms-2
C = 1 / 36.2 # eV-1
I_inf = 3.5e14 * 13.6 # eVs-1m-2
sigma = 6.3e-22 # m2
chi = 0 
alpha = 4e-13 # cm3s-1
B = 0.1
gamma_n = 0.82
z = np.linspace(0, 100000, 1000)
m = 2 * au
mi = au

H = (m * g / (k * T_eff)) # m-1
print('h', 1/H)

# read digitised plots first
DF_fz = pd.read_csv(r'\\uol.le.ac.uk\root\staff\home\d\dsk10\My Documents\Code\Data files\x-z y-f 2.csv', names=['height', 'f'])
DF_pz = pd.read_csv(r'\\uol.le.ac.uk\root\staff\home\d\dsk10\My Documents\Code\Data files\x-z y-p.csv', names=['height', 'p'])
df = pd.merge_ordered(DF_fz, DF_pz, fill_method=None)
df1 = df.copy()
df = df.drop(columns=['f'])
df1 = df1.drop(columns=['p'])
df = df.interpolate(method='nearest')
df1 = df1.interpolate(method='nearest')

def smoothing(x, y):
    y_avg = np.zeros(len(x))
    y_avg[0] = y.iloc[0]
    for i in range(1, len(x)):
        y_avg[i] = (y.iloc[i-1] + y.iloc[i]) / 2
    return y_avg

new_f = smoothing(df['p'], df1['f'])

# now move onto our barometric, isothermal model with reference being P=1.4e-4

P = P0 * np.exp(-z * H)
print('Pmin', P[-1])
n = (P * 1e5) / (k * T_eff) # number density of neutrals in SI

m, c = np.polyfit(z, np.log10(P), 1)

def q():
    #photoionisation on a pressure scale - shouldnt change the profile
    Q = C * sigma * n[0] * I_inf * (P / P0) * np.exp(- (sigma * (1/H) * n[0] ) * (P/P0) / np.cos(chi))
    return Q * 1e-6
print(max(q()))
ions = (q() / alpha) ** 0.5 

f = (ions) / ((n * 1e-6) + ions)

mu_in = ((m * mi) * 1e6) / ((m + mi) * 1e3) / ((1.67e-24)) # reduced mass of ion-neutral pair in atomic units but from grams
nu_ion = 2.6e-9 * (n * 1e-6) * np.sqrt(gamma_n / mu_in)
nu_en = 4.5e-9 * (n * 1e-6) * (1 - (1.35e-4 * T_eff)) / T_eff**0.5

def Omega(z, m):
    return np.full_like(z, e * B / m)

ion_p_contr = (1 / (mi * nu_ion)) * ((nu_ion ** 2) / ((nu_ion ** 2) + (Omega(z, mi) ** 2)))
electron_p_contr = (1 / ((9.1e-31) * nu_en)) * ((nu_en ** 2) / ((nu_en ** 2) + (Omega(z, 9.1e-31) ** 2)))

pedersen = ions * (e ** 2) * (ion_p_contr + electron_p_contr) / 1e-6 # conductivity is in m-3 not cm-3 so had to adjust for that
print('Maximum Pedersen Conductivity:', max(pedersen))

sigma_h = (ions * e ** 2) *(-((1 / (mi * nu_ion)) * ((nu_ion * Omega(z, mi)) / ((nu_ion ** 2) + (Omega(z, mi) ** 2)))) + ((1 / ((9.1e-31) * nu_en)) * ((nu_en * Omega(z, 9.1e-31)) / ((nu_en ** 2) + (Omega(z, 9.1e-31) ** 2))))) / 1e-6

# conductance Sigma_p

con = sum(pedersen * (z[1] - z[0]))
print('Pedersen conductance', con)

gen = (1 / (mi * nu_ion)) + (1 / (9.1e-31 * nu_en)) * ions * e**2

figure, axs = plt.subplots(1, 3, sharey='col')

axs[0].loglog(n * 1e-2, P)
axs[0].set_xlabel(r'Neutral number density, $cm^{-3}$', fontsize=7)
axs[0].set_ylabel('Pressure, bar', fontsize=7)
axs[0].invert_yaxis()
ax0 = axs[0].twinx()
ax0.semilogx(n * 1e-2, z*1e-3)

axs[1].loglog(q(), P, label=rf'$q_m = {max(q()):e}$')
axs[1].set_xlim(5e-4, 5e4)
axs[1].set_xlabel(r'Photoionization rate, $cm^{-3}s^{-1}$', fontsize=7)
axs[1].invert_yaxis()
ax12 = axs[1].twinx()
ax12.semilogx(q(), z*1e-3)
axs[1].set_title(r'Variation of neutral number density, photoionisation rate, and ion number density with height for neutral species of H$_2$ and ionised species of H$^+$ on brown dwarf with log(g)=5', fontsize=8)
axs[1].legend(loc='lower right', prop={'size': 6})

axs[2].loglog(ions, P, label=rf'max $n_i = {int(max(ions)):e}$')
axs[2].set_xlabel(r'Ion number density, $cm^{-3}$', fontsize=7)
axs[2].set_xlim(51000, 1e9)
axs[2].invert_yaxis()
ax13 = axs[2].twinx()
ax13.semilogx(ions, z*1e-3)
ax13.set_ylabel('Altitude, km', fontsize=7)
axs[2].legend(loc='lower right', prop={'size': 6})

plt.tight_layout()
plt.show()

figure, axs = plt.subplots(1, 2, figsize=(10, 5), sharey='col')

axs[0].loglog(Omega(P, mi), P, color='black', label=r'$\Omega_{ci}$')
axs[0].loglog(nu_ion, P, label=r'$\nu_{in}$', color='plum')
axs[0].loglog(nu_en, P, color='darkseagreen', label=r'$\nu_{en}$')
axs[0].set_xlabel(r'log frequency, $s^{-1}$', fontsize=7)
axs[0].legend(loc='lower left')
ax0 = axs[0].twinx()
ax0.semilogx(Omega(P, mi), z*1e-3, color='black')
ax0.semilogx(nu_ion, z*1e-3, color='plum')
ax0.semilogx(nu_en, z*1e-3, color='darkseagreen')
axs[0].invert_yaxis()
axs[0].set_ylabel('Pressure, bar', fontsize=7)

axs[1].loglog(pedersen, P, color='purple', label=rf'$\sigma_p$, maximum $\sigma_p = {max(pedersen):e}$')
axs[1].set_xlabel(r'Conductivity, $\Omega^{-1}m^{-3}$', fontsize=7)
axs[1].loglog(sigma_h, P, color='darkseagreen', label=r'$\sigma_{h}$')
axs[1].loglog(gen, P, color='deeppink', label=r'$\sigma_{0}$')
axs[1].set_xlim(1e-14)
axs[1].legend()
ax1 = axs[1].twinx()
ax1.semilogx(pedersen, z*1e-3, color='purple')
ax1.semilogx(sigma_h, z*1e-3, color='darkseagreen')
ax1.semilogx(gen, z*1e-3, color='deeppink')
axs[1].invert_yaxis()
ax1.set_ylabel('Altitude, km', fontsize=7)
plt.title(rf'Collision frequencies and conductivity of the upper ionosphere on brown dwarf of log(g)=5 for neutral species of H$_2$ and ionised species of H$^+$, conductance={con:2f} mho', x=-0.1, pad=10, fontsize=8)

plt.tight_layout()
plt.show()
