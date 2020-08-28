import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.signal import convolve, convolve2d
from scipy.interpolate import interp1d
from iminuit import Minuit

def diffuse_model(x, c0, theta_d):
    fd = 2*np.pi * np.sin(np.deg2rad(x)) * c0 / theta_d / (x + 0.06*theta_d) * np.exp(-x**2/theta_d**2)
    return fd

def diffuse_model_hpx(nside, c0, theta_d):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, range(npix))
    x = np.rad2deg(theta)
    m_data = c0 / theta_d / (x + 0.06*theta_d) * np.exp(-x**2/theta_d**2)
    return m_data

def PSF(x, sigma):
    return 2*np.pi * np.sin(np.deg2rad(x)) * 1/sigma**2 * np.exp(-x**2/sigma**2/2)

def PSF_hpx(nside, sigma):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, range(npix))
    x = np.rad2deg(theta)
    m_data = 1/sigma**2 * np.exp(-x**2/sigma**2/2)
    return m_data

def convolution_hpx(nside, sigma, c0, theta_d):
    psf = PSF_hpx(nside, sigma)*hp.nside2pixarea(nside)
    diffuse = diffuse_model_hpx(nside, c0, theta_d)*hp.nside2pixarea(nside)
    alm_psf = hp.map2alm(psf, mmax=0)
    alm_diffuse = hp.map2alm(diffuse, mmax=0)
    # np.savetxt('data/alm_psf.dat', alm_psf.T)
    # np.savetxt('data/alm_diffuse.dat', alm_diffuse.T)

    alm_convolution = np.sqrt(4*np.pi / (2*np.arange(3*nside)+1)) * alm_psf * alm_diffuse
    # np.savetxt('data/alm_convolution.dat', alm_convolution.T)
    convolution = hp.alm2map(alm_convolution, nside, lmax=3*nside-1, mmax=0) / hp.nside2pixarea(nside)
    return convolution

def sample_phi(Nsample):
    phi = np.random.uniform(0, 360, Nsample)
    return phi

def sample_diffuse(c0, theta_d):
    N = 1000000
    x = np.random.uniform(0, 15, N)
    y = diffuse_model(x, c0, theta_d)
    u = np.random.uniform(0, 0.004, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data/data_diffuse.dat', np.array([theta, phi]).T)
    return theta, phi

def sample_PSF(sigma):
    N = 1000000
    x = np.random.uniform(0, 2, N)
    y = PSF(x, sigma)
    u = np.random.uniform(0, 0.25, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data/data_PSF.dat', np.array([theta, phi]).T)
    return theta, phi

def transformation(theta_diffuse, phi_diffuse, theta_PSF, phi_PSF):
    theta_PSF = np.deg2rad(theta_PSF)
    phi_PSF = np.deg2rad(phi_PSF)
    theta_diffuse = np.deg2rad(theta_diffuse)
    phi_diffuse = np.deg2rad(phi_diffuse)

    Ry = np.array([[np.cos(theta_diffuse), 0, np.sin(theta_diffuse)], [0, 1, 0], [-np.sin(theta_diffuse), 0, np.cos(theta_diffuse)]])
    Rz = np.array([[np.cos(phi_diffuse), -np.sin(phi_diffuse), 0], [np.sin(phi_diffuse), np.cos(phi_diffuse), 0], [0, 0, 1]])
    vector = np.array([np.sin(theta_PSF)*np.cos(phi_PSF), np.sin(theta_PSF)*np.sin(phi_PSF), np.cos(theta_PSF)]).T
    x, y, z = np.dot(Rz, np.dot(Ry, vector))
    assert abs(x**2+y**2+z**2-1)<1e-9
    theta = np.rad2deg(np.arccos(z))
    phi = np.rad2deg(np.arctan2(y, x))+180
    return theta, phi

def simulation():
    Theta_diffuse, Phi_diffuse = np.loadtxt('data/data_diffuse.dat')[:200000].T
    Theta_PSF, Phi_PSF = np.loadtxt('data/data_PSF.dat')[:200000].T

    theta_conv = []
    phi_conv = []
    #for theta_diffuse, phi_diffuse in zip(Theta_diffuse, Phi_diffuse):
    #    for theta_PSF, phi_PSF in zip(Theta_PSF, Phi_PSF):
    for theta_diffuse, phi_diffuse, theta_PSF, phi_PSF in zip(Theta_diffuse, Phi_diffuse, Theta_PSF, Phi_PSF):
        theta, phi = transformation(theta_diffuse, phi_diffuse, theta_PSF, phi_PSF)
        theta_conv.append(theta)
        phi_conv.append(phi)
    np.savetxt('data/data_convolution.dat', np.array([theta_conv, phi_conv]).T)
    return theta_conv, phi_conv

class Chi2:
    def __init__(self, func, x, y, width):
        self.func = func
        self.x, self.y = x, y
        self.width = width

    def chi2(self, Nargs):
        N, *args = Nargs
        self.f = N * self.func(self.x, *args) * self.width
        chi2 = sum([(y-f)**2 / f for (y, f) in zip(self.y, self.f)])
        #print(Nargs, chi2)
        return chi2
    
    def __call__(self, Nargs):
        return self.chi2(Nargs)

class Lnlike:
    def __init__(self, func, x, y, width):
        self.func = func
        self.x, self.y = x, y
        self.width = width

    def lnlike(self, Nargs):
        N, *args = Nargs
        self.f = N * self.func(self.x, *args) * self.width
        lnlike = sum([y*np.log(f) - f for (y, f) in zip(self.y, self.f)])
        print(Nargs, lnlike)
        return lnlike

    def __call__(self, Nargs):
        return self.lnlike(Nargs)

class Convolution:
    def __init__(self, nside, theta0, ipix0, sigma):
        # x_psf = np.linspace(0, 1, 10)
        # self.psf = PSF(x_psf, sigma)
        self.nside = nside
        self.theta0 = theta0
        self.ipix0 = ipix0
        self.psf_hpx = PSF_hpx(nside, sigma)
        self.alm_psf = hp.map2alm(self.psf_hpx, mmax=0)

    def interp(self, x, c0, theta_d):
        # x_diffuse = np.linspace(0, 10, 100)
        # diffuse = diffuse_model(x_diffuse, c0, theta_d)
        # convolution = convolve(diffuse, self.psf, mode='full')[:len(x_diffuse)]
        # return interp1d(x_diffuse, convolution, kind='cubic')(x)
        diffuse_hpx = diffuse_model_hpx(self.nside, c0, theta_d)
        alm_diffuse = hp.map2alm(diffuse_hpx, mmax=0)
        
        alm_convolution = np.sqrt(4*np.pi / (2*np.arange(3*self.nside)+1)) * self.alm_psf * alm_diffuse
        convolution_hpx = hp.alm2map(alm_convolution, self.nside, lmax=3*self.nside-1, mmax=0)
        convolution0 = convolution_hpx[self.ipix0] * 2*np.pi*np.sin(self.theta0)
        return interp1d(self.theta0, convolution0)(np.deg2rad(x))

    def __call__(self, x, c0, theta_d):
        return self.interp(x, c0, theta_d)

def fit_PSF():
    theta, phi = np.loadtxt('data/data_PSF.dat').T
    print(len(phi))

    plt.figure()
    plt.hist(theta, range=(0, 2), bins=100, density=True)
    plt.xlim(0, 2)
    plt.savefig('data/hist_PSF.png')

    # plt.figure()
    # plt.hist(phi, range=(0, 360), bins=36)
    # plt.xlim(0, 360)
    # plt.show()

    bins = 100
    range_ = (0, 2)
    hist, edges = np.histogram(theta, bins=bins, range=range_)
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1:] - edges[:-1]

    m = Minuit(lambda logN, sigma: -2*Lnlike(PSF, x, hist, width)((10**logN, sigma)), 
    #m = Minuit(lambda N, sigma: Chi2(PSF, x, hist, width)((N, sigma)),
    logN=np.log10(len(theta)), sigma=0.32, error_logN=0.01, error_sigma=0.01, limit_logN=(0, 7), limit_sigma=(0, 0.5))
    m.migrad()
    m.hesse()
    sigma = m.values['sigma']
    error_sigma = m.errors['sigma']

    logN = m.values['logN']
    f = 10**logN * PSF(x, sigma) * width

    plt.figure()
    plt.plot(x, f)
    plt.hist(theta, bins=bins, range=range_)
    plt.xlim(0, 2)
    plt.savefig('data/fitted_PSF.png')

    return sigma, error_sigma

def fit_diffuse():
    theta, phi = np.loadtxt('data/data_diffuse.dat').T
    print(len(phi))

    plt.figure()
    plt.hist(theta, range=(0, 15), bins=100, density=True)
    plt.xlim(0, 15)
    plt.savefig('data/hist_diffuse.png')

    # plt.figure()
    # plt.hist(phi, range=(0, 360), bins=36)
    # plt.xlim(0, 360)
    # plt.show()

    bins = 100
    range_ = (0, 15)
    hist, edges = np.histogram(theta, bins=bins, range=range_)
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1:] - edges[:-1]

    c0 = 1.22 / np.pi**1.5
    m = Minuit(lambda logN, theta_d: -2*Lnlike(diffuse_model, x, hist, width)((10**logN, c0, theta_d)), 
    #m = Minuit(lambda N, theta_d: Chi2(diffuse_model, x, np.log10(hist), width)((N, c0, theta_d)),
    logN=np.log10(len(theta)), theta_d=5, error_logN=0.1, error_theta_d=0.01, limit_logN=(0, 9), limit_theta_d=(0, 10))
    m.migrad()
    m.hesse()
    theta_d = m.values['theta_d']
    error_theta_d = m.errors['theta_d']

    logN = m.values['logN']
    f = 10**logN * diffuse_model(x, c0, theta_d) * width

    plt.figure()
    plt.plot(x, f)
    plt.hist(theta, bins=bins, range=range_)
    plt.xlim(0, 15)
    # plt.yscale('log')
    plt.savefig('data/fitted_diffuse.png')

    return theta_d, error_theta_d

def fit_convolution():
    theta, phi = np.loadtxt('data/data_convolution.dat').T
    theta0 = np.loadtxt('theta0.dat').T
    ipix0 = np.loadtxt('ipix0.dat').T.astype('int')
    print(len(phi))
    # theta = np.loadtxt('data2.txt')[:, 4]

    plt.figure()
    plt.hist(theta, range=(0, 15), bins=100, density=True)
    plt.xlim(0, 15)
    # plt.yscale('log')
    plt.savefig('data/hist_convolution.png')

    # plt.figure()
    # plt.xlim(0, 360)
    # plt.hist(phi, range=(0, 360), bins=36)
    # plt.show()

    bins = 100
    range_ = (0, 15)
    hist, edges = np.histogram(theta, bins=bins, range=range_)
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1:] - edges[:-1]

    c0 = 1.22 / np.pi**1.5
    sigma = 0.3
    nside = 1024

    m = Minuit(lambda logN, theta_d: -2*Lnlike(Convolution(nside, theta0, ipix0, sigma), x, hist, width)((10**logN, c0, theta_d)), logN=np.log10(len(theta)), theta_d=5, error_logN=0.01, error_theta_d=0.01, limit_logN=(0, 99), limit_theta_d=(0, 10))
    m.migrad()
    m.hesse()
    theta_d = m.values['theta_d']
    error_theta_d = m.errors['theta_d']

    logN = m.values['logN']
    f = 10**logN * Convolution(nside, theta0, ipix0, sigma)(x, c0, theta_d) * width

    plt.figure()
    plt.plot(x, f)
    plt.hist(theta, bins=bins, range=range_)
    plt.xlim(0, 15)
    # plt.yscale('log')
    plt.savefig('data/fitted_convolution.png')

    return theta_d, error_theta_d


if __name__ == "__main__":
    # from collections import Counter
    nside = 1024
    nring = 4*nside - 1
    npix = hp.nside2npix(nside)
    # theta, phi = hp.pix2ang(nside, range(npix))
    # counter = Counter(theta)
    # ipix = np.cumsum(list(counter.values())) - 1
    # theta0 = theta[ipix]
    # phi0 = phi[ipix]
    # filt = theta0 < np.deg2rad(15)
    # theta0 = theta0[filt]
    # phi0 = phi0[filt]
    # ipix0 = hp.ang2pix(nside, theta0, phi0)
    # np.savetxt('theta0.dat', theta0.T)
    # np.savetxt('ipix0.dat', ipix0.T, fmt='%i')

    theta0 = np.loadtxt('theta0.dat').T
    ipix0 = np.loadtxt('ipix0.dat').T.astype('int')

    sigma = 0.3
    x_psf = np.linspace(0, 2, 100)
    psf = PSF(x_psf, sigma)
    psf_hpx = PSF_hpx(nside, sigma)
    psf0 = psf_hpx[ipix0] * 2*np.pi * np.sin(theta0)

    # print(psf_hpx)
    # np.savetxt('data/hpx_PSF.dat', psf_hpx.T)
    # hp.mollview(psf_hpx, rot=(0, 90, 0))
    # plt.savefig('data/hpx_PSF.png')

    plt.figure()
    plt.plot(np.rad2deg(theta0), psf0, 'o')
    plt.plot(x_psf, psf)
    plt.xlim(0, 2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{dN}{d\theta}$')
    plt.savefig('dN_dTheta_PSF.png')

    plt.figure()
    plt.plot(np.rad2deg(theta0), psf0/np.sin(theta0), 'o')
    plt.plot(x_psf, psf/np.sin(np.deg2rad(x_psf)))
    plt.xlim(0, 2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{dN}{d\Omega}$')
    plt.savefig('dN_dOmega_PSF.png')

    c0 = 1.22 / np.pi**1.5
    theta_d = 5.5
    x_diffuse = np.linspace(0, 15, 500)
    diffuse = diffuse_model(x_diffuse, c0, theta_d)
    diffuse_hpx = diffuse_model_hpx(nside, c0, theta_d)
    diffuse0 = diffuse_hpx[ipix0]*2*np.pi*np.sin(theta0)

    # print(diffuse_hpx)
    # np.savetxt('data/hpx_diffuse.dat', diffuse_hpx.T)
    # hp.mollview(diffuse_hpx, rot=(0, 90, 0))
    # plt.savefig('data/hpx_diffuse.png')

    plt.figure()
    plt.plot(np.rad2deg(theta0), diffuse0, 'o')
    plt.plot(x_diffuse, diffuse)
    plt.xlim(0, 15)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{dN}{d\theta}$')
    plt.savefig('dN_dTheta_diffuse.png')

    plt.figure()
    plt.plot(np.rad2deg(theta0), diffuse0/np.sin(theta0), 'o')
    plt.plot(x_diffuse, diffuse/np.sin(np.deg2rad(x_diffuse)))
    plt.xlim(0, 15)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{dN}{d\Omega}$')
    plt.savefig('dN_dOmega_diffuse.png')

    convolution_hpx = convolution_hpx(nside, sigma, c0, theta_d)
    convolution0 = convolution_hpx[ipix0] * 2*np.pi*np.sin(theta0)
    # print(convolution_hpx)
    # np.savetxt('data/hpx_convolution.dat', convolution_hpx.T)
    # hp.mollview(convolution_hpx, rot=(0, 90, 0))
    # plt.savefig('data/hpx_convolution.png')

    np.savetxt('dist_convolution.dat', np.array([np.rad2deg(theta0), convolution0]).T)

    plt.figure()
    plt.plot(np.rad2deg(theta0), convolution0, 'o')
    plt.xlim(0, 15)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{dN}{d\theta}$')
    plt.savefig('dN_dTheta_convolution.png')

    plt.figure()
    plt.plot(np.rad2deg(theta0), convolution0/np.sin(theta0), 'o')
    plt.xlim(0, 15)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{dN}{d\Omega}$')
    plt.savefig('dN_dOmega_convolution.png')

    # sample_PSF(sigma)
    fitted_sigma, error_sigma = fit_PSF()
    print('Fitting PSF result:', fitted_sigma, error_sigma)

    # sample_diffuse(c0, theta_d)
    fitted_theta_d, error_theta_d = fit_diffuse()
    print('Fitting diffuse model result:', fitted_theta_d, error_theta_d)

    # simulation()
    fitted_theta_d, error_theta_d = fit_convolution()
    print('Fitting convolution result:', fitted_theta_d, error_theta_d)