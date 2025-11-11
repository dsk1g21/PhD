import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('Starting the code')
M_B = 1.898e27 * 21
R_B = 71492e3
G = 6.6743e-11
g = G * M_B / R_B ** 2 # ms-2
print('Gravity', g)
T_eff = 780 # K
logg = 3.0
g_B = 1.2 # Gaunt factor
n0 = 3e14
alpha = 4e-13 # cm3s-1
alpha_1 =  alpha # * 0.38
sigma = 6.3e-18 # cm2 ionisation cross section
sigma_gas = 8.8e-17 # cm2
B_background = 0.1 # T
chi = 0
h = 4.135e-15
E = 18 * h * 2.998e8 / (688e-10)
I_inf = 3.5e14 * 13.6 #evs-1m-2
C = 1/36.2
k = 1.38e-23 # m2kgs-2K-18
e = 1.6e-19 # C to keep B in Teslas
gamma_n = 0.82e-24 # cm3
H2 = 2
H1 = 1
au = 1.67e-27
m = H2 * au
mi = H1 * au

df = pd.read_csv(r'Data files\plot-data.csv', names=['height', 'pressure'])

DF_fz = pd.read_csv(r'\\uol.le.ac.uk\root\staff\home\d\dsk10\My Documents\Code\Data files\x-z y-f.csv', names=['height', 'f'])
DF_pz = pd.read_csv(r'\\uol.le.ac.uk\root\staff\home\d\dsk10\My Documents\Code\Data files\plot-data.csv', names=['height', 'p'])

df = pd.merge_ordered(DF_fz, DF_pz, fill_method=None)
#df = df.sort_values(by='height', ascending=False)
df1 = df.copy()
df = df.drop(columns=['f'])
df1 = df1.drop(columns=['p'])
df = df.interpolate(method='quadratic')
df1 = df1.interpolate(method='quadratic')
#df.reindex(index=df.index[::-1])
print(df)
def smoothing(x, y):
    y_avg = np.zeros(len(x))
    y_avg[0] = y.iloc[0]
    for i in range(1, len(x)):
        y_avg[i] = (y.iloc[i-1] + y.iloc[i]) / 2
    return y_avg

new_f = smoothing(df['p'], df1['f'])

z = np.linspace(0, 4000000, 10000)
df['height'] = df['height'] 
H = m * g / (k * T_eff) # ~7km 
print('Scale height', 1/H)
P = np.exp(-z * H)
def q(z):
    Q = C * sigma * n0 * I_inf * np.exp(-z*H - (sigma * (1/H) * n0 ) * np.exp(-z*H) / np.cos(chi))
    return Q * 1e-6

neutrals = n0 * np.exp(-z * H)
print('max photoionisation', max(q(z)))
ions = ((q(z))/alpha_1)**0.5

f = ions / (neutrals + ions)
print('max number ions', max(ions))

mu_in = ((m * mi) * 1e6) / ((m + mi) * 1e3) / ((1.67e-24)) # reduced mass of ion-neutral pair in atomic units but from grams
print('Reduced mass', mu_in)

ee = 4.803205e-10

nu_ion = 2.6e-9 * (neutrals * 1e-6) * np.sqrt(0.82 / mu_in)
nu_en = 4.5e-9 * (neutrals) * (1 - (1.35e-4 * T_eff)) / T_eff**0.5
print('collision frequency', (nu_ion))
Omega_ci = np.zeros(len(z))
Omega_ce = np.zeros(len(z))
Omega_ci.fill(((e * B_background / mi))) 
Omega_ce.fill(((e * B_background) / 9.1e-31))
omega_ci = np.sqrt(ions * e**2 / (mi * 8.854187817e-12))

ion_p_contr = (1 / (mi * nu_ion)) * ((nu_ion ** 2) / ((nu_ion ** 2) + (Omega_ci ** 2)))
electron_p_contr = (1 / ((9.1e-31) * nu_en)) * ((nu_en ** 2) / ((nu_en ** 2) + (Omega_ce ** 2)))

pedersen = ions * (e ** 2) * (ion_p_contr + electron_p_contr) / 1e-6 # conductivity is in m-3 not cm-3 so had to adjust for that
print('Maximum Pedersen Conductivity:', max(pedersen))

sigma_h = (ions * e ** 2) *(-((1 / (mi * nu_ion)) * ((nu_ion * Omega_ci) / ((nu_ion ** 2) + (Omega_ci ** 2)))) + ((1 / ((9.1e-31) * nu_en)) * ((nu_en * Omega_ce) / ((nu_en ** 2) + (Omega_ce ** 2))))) / 1e-6

# conductance Sigma_p

con = sum(pedersen * (z[1] - z[0]))
print('Pedersen Conductance:', con)
plt.ylabel(r'f')
plt.loglog(P, f, label='Initial number density=3e14')
plt.loglog(df['p'], df1['f'])
plt.xlabel('pressure')
plt.ylim(1e-12, 1e0)
plt.xlim(1e-14, 1e0)
plt.legend()
plt.gca().invert_xaxis()

plt.show()

fig, axs = plt.subplots(1, 3, sharey='row')
axs[0].semilogx(neutrals * 1e-6, z * 1e-3)
axs[0].set_xlim(1e-50, n0)
axs[0].set_xlabel(r'Neutral number density, $cm^{-3}$', fontsize=7)
axs[0].set_title('Neutral number density variation with altitude', fontsize=7)
axs[1].semilogx(q(z), z*1e-3)
axs[1].set_xlabel(r'Photoionization rate, $cm^{-3}s^{-1}$', fontsize=7)
axs[1].set_title('Photoionization variation with altitude', fontsize=7)
axs[1].set_xlim(0.01, 1e6)
axs[1].set_ylim(-10, 1000)
axs[2].semilogx(ions, z*1e-3)
axs[2].set_xlabel(r'Ion number density, $cm^{-3}$', fontsize=7)
axs[2].set_title('Ion density variation with altitude', fontsize=7)
axs[2].set_xlim(1e2)
axs[2].set_ylim(-10, 1000)
plt.ylabel(r'Reduced Height, km', fontsize=7)
plt.tight_layout()
plt.show()

plt.semilogy(z, P)
plt.ylabel(r'P')
#plt.loglog(P, f)
plt.ylim(1e-14, 1e2)
plt.xlabel('pressure')
#plt.gca().invert_xaxis()

figure, axs = plt.subplots(1, 2, figsize=(10, 5), sharey='row')

axs[0].semilogx(Omega_ci, z * 1e-3, color='black', label=r'$\Omega_{ci}$')
axs[0].semilogx(nu_ion, z * 1e-3, label=r'$\nu_{in}$', color='plum')
axs[0].semilogx(nu_en, z * 1e-3, color='darkseagreen', label=r'$\nu_{en}$')
axs[0].set_xlabel(r'log frequency, $s^{-1}$')
axs[0].legend(loc='lower left')
axs[1].semilogx(pedersen, z * 1e-3, color='purple', label=r'$\sigma_{p}$')
axs[1].set_xlabel(r'Conductivity, $\Omega^{-1}m^{-3}$')
axs[1].set_xlim(1e-30)
axs[1].legend()
plt.ylim(-10, 1000)
plt.ylabel('altitude, km')
plt.show()

