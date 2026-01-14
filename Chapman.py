import numpy as np
import matplotlib.pyplot as plt

class Chapman:
    k = 1.38e-23 # Boltzmann constant, m2kgs-2K-1
    e = 1.6e-19 # Electron charge, C
    au = 1.67e-27 # Atomic mass unit, kg
    me = 9.1e-31 # Electron mass, kg

    def __init__(self, name, T, g, P0, m_ion, m_neutral, z, B, C, I_inf, n0, chi=0):
        self.name = name
        self.T = T
        self.g = g
        self.P0 = P0
        self.chi = chi
        self.z = np.linspace(0, z, 10000)
        self.B = B
        self.C = C
        self.I_inf = I_inf
        self.n0 = n0
        if m_neutral == 'H':
            self.m_neutral = 1
        elif m_neutral == 'H2':
            self.m_neutral = 2
        elif m_neutral == 'O':
            self.m_neutral = 16
        elif m_neutral == 'O2':
            self.m_neutral = 32
        elif m_neutral == 'N2':
            self.m_neutral = 28
        else:
            raise ValueError('Neutrals species not yet added')
        if m_ion == 'H+':
            self.m_ion = 1
        elif m_ion == 'H3+':
            self.m_ion = 3
        elif m_ion == 'O2+':
            self.m_ion = 32
        elif m_ion == 'N+':
            self.m_ion = 14
        else:
            raise ValueError('Ion species not yet added')
        if m_ion == 'H+':
            self.sigma = 6.3e-22
        elif m_ion == 'O2+':
            self.sigma = 20e-22
        else:
            raise ValueError('Ion species not yet added')
        
    def alpha(self):
        if self.m_ion == 1:
            return 4e-13
        elif self.m_ion == 14:
            return (1.8e-7* (300/self.T) ** 0.39)
        elif self.m_ion == 32:
            return (2.4e-7* (300/self.T) ** 0.7)
        elif self.m_ion == 3:
            return 1.15e-7 * (300/self.T) ** 0.65
        else:
            raise ValueError('Neutral species not yet added')

    def mass(self):
        return self.au * self.m_neutral, self.au * self.m_ion

    def mu_in(self):
        m, mi = self.mass()
        return ((m * mi) * 1e6) / ((m + mi) * 1e3) / ((1.67e-24)) # reduced mass of ion-neutral pair in atomic units but from grams

    def scale_height(self):
        # scale height in m-1 so for height print 1/scale_height
        m, mi = self.mass()
        return m * self.g / (self.k * self.T)

    def pressure_scale(self):
      # bars
      return self.P0 * np.exp( - self.z * self.scale_height())

    def neutrals(self):
        if self.n0 != 0:
            return self.n0 * np.exp( - self.z * self.scale_height())
        else:
            return (self.pressure_scale() * 1e5) / (self.k * self.T)

    def q(self):
        Q = self.C * self.sigma * self.neutrals()[0] * self.I_inf * np.exp(-self.z * self.scale_height() 
                                                            - (self.sigma * (1/self.scale_height()) * self.neutrals()[0]) 
                                                           * np.exp(-self.z * self.scale_height()) / np.cos(self.chi))
        return Q * 1e-6 # cm-3

    def ions(self):
        return (self.q() / self.alpha()) ** 0.5 # cm-3
    
    def nu_in(self):
        if self.m_neutral == 2:
            gamma_n = 0.82
            return 2.6e-9 * self.neutrals() * 1e-6 * np.sqrt(gamma_n / self.mu_in())
        elif self.m_neutral == 28:
            gamma_n = 1.76
            return 2.6e-9 * self.neutrals() * 1e-6 * np.sqrt(gamma_n / self.mu_in())
        elif self.m_neutral == 16:
            gamma_n = 0.77
            return 2.6e-9 * self.neutrals() * 1e-6 * np.sqrt(gamma_n / self.mu_in())
        else:
            raise ValueError('Neutral species yet to be added')
    
    def nu_en(self):
        if self.m_neutral == 2:
            return 3.31e-8 * self.neutrals() * 1e-6
        elif self.m_neutral == 28:
            return 2.33e-11 * self.neutrals() * 1e-6 * (1 - 1.21e-4 * self.T) * self.T
        else:
            raise ValueError('Neutral species yet to be added')
    
    def Omega(self):
        # returns ion gyrofrequency as [0] and electron gyrofrequency as [1]
        main_component = self.e * self.B
        return np.full_like(self.z, main_component / self.mass()[1]), np.full_like(self.z, main_component / self.me)
    
    def conductivity(self):
        # returns general conductivity as [0], hall as [1] and pedersen as [2]
        N_nu = (self.e ** 2) * self.ions() / 1e-6
        den_i = self.mass()[1] * self.nu_in()
        den_e = self.me * self.nu_en()
        sigma_0 = (1 / (self.mass()[1] * self.nu_in())) + (1 / (9.1e-31 * self.nu_en())) * self.ions() * self.e**2 /1e-6
        sigma_p = N_nu * ((1/den_i * (self.nu_in() ** 2 / (self.nu_in() ** 2 + self.Omega()[0] ** 2)))
                           + (1/den_e * (self.nu_en() ** 2 / (self.nu_en() ** 2 + self.Omega()[1] ** 2))))
        sigma_h = N_nu * ((1/den_e * (self.nu_en() * self.Omega()[1] / (self.nu_en() ** 2 + self.Omega()[1] ** 2)))
                           - (1/den_i * (self.nu_in() * self.Omega()[0] / (self.nu_in() ** 2 + self.Omega()[0] ** 2))))
        return sigma_0, sigma_h, sigma_p

    def conductance(self):
        return sum((self.z[1] - self.z[0]) * self.conductivity()[2])

    def plot_numberdensity(self):
        figure, axs = plt.subplots(1, 3, sharey='col', figsize=(10, 5))

        axs[0].loglog(self.neutrals() * 1e-2, self.pressure_scale())
        axs[0].set_xlabel(r'Neutral number density, $cm^{-3}$', fontsize=7)
        axs[0].set_ylabel('Pressure, bar', fontsize=7)
        axs[0].invert_yaxis()
        ax0 = axs[0].twinx()
        ax0.semilogx(self.neutrals() * 1e-2, self.z*1e-3)

        axs[1].loglog(self.q(), self.pressure_scale(), label=rf'$q_m = {max(self.q()):e}$')
        axs[1].set_xlim((max(self.q()) / 1e6), (max(self.q()) * 1e3))
        axs[1].set_xlabel(r'Photoionization rate, $cm^{-3}s^{-1}$', fontsize=7)
        axs[1].invert_yaxis()
        ax12 = axs[1].twinx()
        ax12.semilogx(self.q(), self.z*1e-3)
        axs[1].set_title(r'Variation of neutral number density, photoionisation rate, and ion number density with height', fontsize=8)
        axs[1].legend(loc='lower right', prop={'size': 6})

        axs[2].loglog(self.ions(), self.pressure_scale(), label=rf'max $n_i = {int(max(self.ions())):e}$')
        axs[2].set_xlabel(r'Ion number density, $cm^{-3}$', fontsize=7)
        axs[2].set_xlim((max(self.ions()) / max(self.ions())), (max(self.ions()) * 1e3))
        axs[2].invert_yaxis()
        ax13 = axs[2].twinx()
        ax13.semilogx(self.ions(), self.z*1e-3)
        ax13.set_ylabel('Altitude, km', fontsize=7)
        axs[2].legend(loc='lower right', prop={'size': 6})

        plt.tight_layout()

    def plot_conductivity(self):
        
        figure, axs = plt.subplots(1, 2, figsize=(10, 5), sharey='col')

        axs[0].loglog(self.Omega()[0], self.pressure_scale(), color='black', label=r'$\Omega_{ci}$')
        axs[0].loglog(self.nu_in(), self.pressure_scale(), label=r'$\nu_{in}$', color='plum')
        axs[0].loglog(self.nu_en(), self.pressure_scale(), color='darkseagreen', label=r'$\nu_{en}$')
        axs[0].set_xlabel(r'log frequency, $s^{-1}$', fontsize=7)
        axs[0].legend(loc='lower left')
        ax0 = axs[0].twinx()
        ax0.semilogx(self.Omega()[0], self.z*1e-3, color='black')
        ax0.semilogx(self.nu_in(), self.z*1e-3, color='plum')
        ax0.semilogx(self.nu_en(), self.z*1e-3, color='darkseagreen')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Pressure, bar', fontsize=7)

        axs[1].loglog(self.conductivity()[2], self.pressure_scale(), color='purple', label=rf'$\sigma_p$, maximum $\sigma_p = {max(self.conductivity()[2]):e}$')
        axs[1].set_xlabel(r'Conductivity, $\Omega^{-1}m^{-3}$', fontsize=7)
        axs[1].loglog(self.conductivity()[1], self.pressure_scale(), color='darkseagreen', label=r'$\sigma_{h}$')
        axs[1].loglog(self.conductivity()[0], self.pressure_scale(), color='deeppink', label=r'$\sigma_{0}$')
        axs[1].set_xlim(1e-20)
        axs[1].legend()
        ax1 = axs[1].twinx()
        ax1.semilogx(self.conductivity()[2], self.z*1e-3, color='purple')
        ax1.semilogx(self.conductivity()[1], self.z*1e-3, color='darkseagreen')
        ax1.semilogx(self.conductivity()[0], self.z*1e-3, color='deeppink')
        axs[1].invert_yaxis()
        ax1.set_ylabel('Altitude, km', fontsize=7)
        plt.title(rf'Collision frequencies and conductivity of the upper ionosphere, conductance={self.conductance():e} mho', x=-0.1, pad=10, fontsize=8)

        plt.tight_layout()

    @classmethod
    def planetary_object(cls, name, **kwargs):
        name = name.lower()
        G = 6.6743e-11 # gravitational constant
        if name == 'earth':
            return cls("Earth", T=600, g=9.81, C=1/35, m_ion='O2+', m_neutral='N2', B=60000e-9, **kwargs)
        elif name == 'jupiter':
            M_J = 1.898e27
            R_J = 71492e3
            g = G * M_J / R_J ** 2
            return cls("Jupiter", 1000, g=g, C=1/36.2, B=834000e-9, m_neutral='H2', **kwargs)
        elif 'brown dwarf' in name:
            return cls("Brown dwarf", m_neutral='H2', C=1/36.2, B=0.05, **kwargs)
        else:
            raise ValueError(f"Unknown object: {name}")
