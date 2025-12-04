import numpy as np
import matplotlib.pyplot as plt

# This model is only for H+ ionisation
print('Starting the code')
M_J = 1.898e27
R_J = 71492e3
G = 6.6743e-11
n0 = 1e20 # m-3
Te = 1000 # K
alpha = 4.8e-12 * (250 / Te)**0.7 # cm-3
g = G * M_J / R_J ** 2 # ms-2
print('gravity', g)
k = 1.38e-23 # m2kgs-2K-1
sigma = 6.3e-22 # m2
C = 1/36.2 # eV-1
I_inf =  1e16 / 25 # eVm-2s-1
chi = 0
e = 1.6e-19 # C to keep B in Teslas
gamma_n = 0.82 # x10^-24 cm3
B = 834e-6 # T
H1 = 1
H2 = 2 

def mass(n):
    return n * 1.67e-27 # returns kg

m = mass(H2)
h = 10000

H = (m * g) / (k * Te) # m-1 ~166km
print('scale height', 1/H)

z = np.linspace(0, 5000000, h+1)

def q(z):
    Q = C * sigma * n0 * I_inf * np.exp(-z*H - (sigma * (1/H) * n0 ) * np.exp(-z*H) / np.cos(chi))
    return Q * 1e-6

Q_max = (C * I_inf * np.cos(chi)) / ((1/H) * np.exp(1))
print('maximum photoionisation', Q_max * 1e-6)

neutrals = n0 * np.exp(-z * H)

ions = ((q(z))/alpha)**0.5
print('initial ion density', ions[0])

mi = mass(H1)

print('maximum ion density', max(ions))

mu_in = ((m * mi) * 1e6) / ((m + mi) * 1e3) / ((1.67e-24)) # reduced mass of ion-neutral pair in atomic units but from grams
print('Reduced mass', mu_in)

ee = 4.803205e-10

nu_ion = 2.6e-9 * (neutrals * 1e-6) * np.sqrt(gamma_n / mu_in)
nu_en = 3.31e-8 * neutrals * 1e-6 #4.5e-9 * (neutrals) * (1 - (1.35e-4 * Te)) / Te**0.5
print('collision frequency', (nu_ion))
Omega_ci = np.zeros(len(z))
Omega_ce = np.zeros(len(z))
Omega_ci.fill(((e * B / mi))) 
Omega_ce.fill(((e * B) / 9.1e-31))
print('omega', Omega_ce[1])
ion_p_contr = (1 / (mi * nu_ion)) * ((nu_ion ** 2) / ((nu_ion ** 2) + (Omega_ci ** 2)))
electron_p_contr = (1 / ((9.1e-31) * nu_en)) * ((nu_en ** 2) / ((nu_en ** 2) + (Omega_ce ** 2)))

pedersen = ions * (e ** 2) * (ion_p_contr + electron_p_contr) / 1e-6 # conductivity is in m-3 not cm-3 so had to adjust for that
print(max(pedersen))
'''for i in range(len(z)):
    if pedersen[i] == max(pedersen):
        print('max height:', z[i])'''
sigma_h = (ions * e ** 2) *(-((1 / (mi * nu_ion)) * ((nu_ion * Omega_ci) / ((nu_ion ** 2) + (Omega_ci ** 2)))) + ((1 / ((9.1e-31) * nu_en)) * ((nu_en * Omega_ce) / ((nu_en ** 2) + (Omega_ce ** 2))))) / 1e-6

# conductance Sigma_p
con = sum(pedersen * (z[1] - z[0]))
print('con', con)

figure, axs = plt.subplots(1, 3, sharey='row', figsize=(10, 5))
axs[0].semilogx(neutrals * 1e-6, z * 1e-3)
axs[0].set_xlabel(r'Neutral number density, $cm^{-3}$', fontsize=7)
axs[0].set_ylabel(r'Reduced Height, km', fontsize=7)
#axs[0].set_title('Neutral number density variation with altitude', fontsize=7)
axs[1].semilogx(q(z), z*1e-3, label=f'Peak ionisation={int(max(q(z)))} at z=1543.5km')
axs[1].set_xlabel(r'Photoionization rate, $cm^{-3}s^{-1}$', fontsize=7)
axs[1].legend(loc='lower left', prop={'size': 6})
axs[1].set_xlim(1e-3, 1e3)
#axs[1].set_title('Photoionization variation with altitude', fontsize=7)
axs[1].set_title(r'Variation of neutral number density, photoionisation rate, and ion number density with height for neutral species of H$_2$ and ionised species of H$^+$', fontsize=8)
axs[2].semilogx(ions, z*1e-3, label=f'Peak ion density={int(max(ions))} at z=106.3km')
axs[2].set_xlabel(r'Ion number density, $cm^{-3}$', fontsize=7)
axs[2].legend(loc='lower left', prop={'size': 6})
axs[2].set_xlim(1e3, 2e7)
#axs[2].set_title('Ion density variation with altitude', fontsize=7)
plt.tight_layout()

figure, axs = plt.subplots(1, 2, figsize=(10, 5), sharey='row')

axs[0].semilogx(Omega_ci, z * 1e-3, color='black', label=r'$\Omega_{ci}$')
axs[0].semilogx(nu_en, z * 1e-3, color='darkseagreen', label=r'$\nu_{en}$')
axs[0].semilogx(nu_ion, z * 1e-3, label=r'$\nu_{in}$', color='plum')
axs[0].set_xlabel(r'log frequency, $s^{-1}$')
axs[0].legend(loc='lower left')
axs[1].semilogx(pedersen, z * 1e-3, color='purple', label=fr'$\sigma_p$, peak at 1360.5km, maximum $\sigma_p = {max(pedersen):e}$ mho/m')
axs[1].semilogx(sigma_h, z * 1e-3, color='darkseagreen', label=r'$\sigma_h$')
axs[1].set_xlabel(r'Conductivity, $\Omega^{-1}m^{-3}$')
axs[1].set_xlim(8e-13, 1e-5)
axs[1].legend(loc='upper right', prop={'size': 6})
plt.title(rf'Collision frequencies and conductivity of the upper ionosphere on Jupiter for neutral species of H$_2$ and ionised species of H$^+$, conductance={con:2f} mho', x=-0.1, pad=10, fontsize=8)
plt.ylabel('altitude, km')
plt.show()
