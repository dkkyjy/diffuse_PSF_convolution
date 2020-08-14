import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.signal import convolve, fftconvolve
from scipy.interpolate import interp1d
from iminuit import Minuit

def diffuse_model(x, c0, theta_d):
    fd = c0 / theta_d / (x + 0.06*theta_d) * np.exp(-x**2/theta_d**2)
    return fd

def PSF_1D(x, sigma):
    # return norm.pdf(x, scale=sigma)
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-x**2/sigma**2/2)

def PSF_2D(x, y, sigma):
    # return multivariate_normal.pdf([x, y], cov=width)
    return 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/sigma**2/2)

def PSF(x, sigma):
    return 1/sigma**2 * np.exp(-x**2/sigma**2/2)

def convolution(x_diffuse, c0, theta_d, x_psf, sigma):
    model = diffuse_model(x_diffuse, c0, theta_d)
    psf = PSF(x_psf, sigma)
    convolution = fftconvolve(model, psf, mode='full')[:len(x_diffuse)]
    return convolution

def sample_phi(Nsample):
    phi = np.random.uniform(0, 360, Nsample)
    return phi

def sample_diffuse(c0, theta_d):
    N = 100000000
    x = np.random.uniform(0, 10, N)
    y = diffuse_model(x, c0, theta_d)
    u = np.random.uniform(0, 1.2, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data_diffuse.dat', np.array([theta, phi]).T)
    return theta, phi

def sample_PSF(sigma):
    N = 100000
    x = np.random.uniform(0, 1, N)
    y = PSF(x, sigma)
    u = np.random.uniform(0, 12, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data_PSF.dat', np.array([theta, phi]).T)
    return theta, phi

def transformation(theta_diffuse, phi_diffuse, theta_PSF, phi_PSF):
    theta_PSF = np.deg2rad(theta_PSF)
    phi_PSF = np.deg2rad(phi_PSF)
    theta_diffuse = np.deg2rad(theta_diffuse)
    phi_diffuse = np.deg2rad(phi_diffuse)

    Ry = np.array([[np.cos(theta_diffuse), 0, -np.sin(theta_diffuse)], [0, 1, 0], [np.sin(theta_diffuse), 0, np.cos(theta_diffuse)]])
    Rz = np.array([[np.cos(phi_diffuse), -np.sin(phi_diffuse), 0], [np.sin(phi_diffuse), np.cos(phi_diffuse), 0], [0, 0, 1]])
    vector = np.array([np.sin(theta_PSF)*np.cos(phi_PSF), np.sin(theta_PSF)*np.sin(phi_PSF), np.cos(theta_PSF)]).T
    x, y, z = np.dot(Rz, np.dot(Ry, vector))
    assert abs(x**2+y**2+z**2-1)<1e-9
    theta = np.rad2deg(np.arccos(z))
    phi = np.rad2deg(np.arctan2(y, x))+180
    return theta, phi

def simulation():
    Theta_diffuse, Phi_diffuse = np.loadtxt('data_diffuse.dat')[:1000].T
    Theta_PSF, Phi_PSF = np.loadtxt('data_PSF.dat')[:1000].T

    theta_conv = []
    phi_conv = []
    for theta_diffuse, phi_diffuse in zip(Theta_diffuse, Phi_diffuse):
        for theta_PSF, phi_PSF in zip(Theta_PSF, Phi_PSF):
            theta, phi = transformation(theta_diffuse, phi_diffuse, theta_PSF, phi_PSF)
            theta_conv.append(theta)
            phi_conv.append(phi)
    np.savetxt('data_convolution.dat', np.array([theta_conv, phi_conv]).T)
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
        #print(Nargs, lnlike)
        return lnlike

    def __call__(self, Nargs):
        return self.lnlike(Nargs)

class Convolution:
    def __init__(self, sigma):
        x_psf = np.linspace(0, 1, 10)
        self.psf = PSF(x_psf, sigma)

    def interp(self, x, c0, theta_d):
        x_diffuse = np.linspace(0, 10, 100)
        model = diffuse_model(x_diffuse, c0, theta_d)
        convolution = fftconvolve(model, self.psf, mode='full')[:len(x_diffuse)]
        return interp1d(x_diffuse, convolution, kind='cubic')(x)

    def __call__(self, x, c0, theta_d):
        return self.interp(x, c0, theta_d)

def fit_PSF():
    theta, phi = np.loadtxt('data_PSF.dat').T

    # plt.figure()
    # plt.hist(theta, range=(0, 1), bins=10)
    # plt.xlim(0, 1)
    # plt.show()

    # plt.figure()
    # plt.hist(phi, range=(0, 360), bins=36)
    # plt.xlim(0, 360)
    # plt.show()

    bins = 10
    range = (0, 1)
    hist, edges = np.histogram(theta, bins=bins, range=range)
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1:] - edges[:-1]

    m = Minuit(lambda N, sigma: -2*Lnlike(PSF, x, hist, width)((N, sigma)), 
    #m = Minuit(lambda N, sigma: Chi2(PSF, x, hist, width)((N, sigma)),
    N=len(theta), sigma=0.32, error_N=0.1, error_sigma=0.01, limit_N=(0, 100000), limit_sigma=(0, 0.5))
    m.migrad()
    m.hesse()
    sigma = m.values['sigma']
    error_sigma = m.errors['sigma']

    # N = m.values['N']
    # f = N * PSF(x, sigma) * width
    # plt.figure()
    # plt.plot(x, f)
    # plt.hist(theta, bins=bins, range=range)
    # plt.xlim(0, 1)
    # plt.savefig('fitted_PSF.png')

    return sigma, error_sigma

def fit_diffuse():
    theta, phi = np.loadtxt('data_diffuse.dat').T

    # plt.figure()
    # plt.hist(theta, range=(0, 10), bins=10)
    # plt.xlim(0, 10)
    # plt.show()

    # plt.figure()
    # plt.hist(phi, range=(0, 360), bins=36)
    # plt.xlim(0, 360)
    # plt.show()

    bins = 100
    range = (0, 10)
    hist, edges = np.histogram(theta, bins=bins, range=range)
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1:] - edges[:-1]

    c0 = 1.22 / np.pi**1.5
    m = Minuit(lambda N, theta_d: -2*Lnlike(diffuse_model, x, hist, width)((N, c0, theta_d)), 
    #m = Minuit(lambda N, theta_d: Chi2(diffuse_model, x, np.log10(hist), width)((N, c0, theta_d)),
    N=len(theta), theta_d=5, error_N=0.1, error_theta_d=0.01, limit_N=(0, 1e7), limit_theta_d=(0, 10))
    m.migrad()
    m.hesse()
    theta_d = m.values['theta_d']
    error_theta_d = m.errors['theta_d']

    # N = m.values['N']
    # f = N * diffuse_model(x, c0, theta_d) * width

    # plt.figure()
    # plt.plot(x, f)
    # plt.hist(theta, bins=bins, range=range)
    # plt.xlim(0, 10)
    # plt.yscale('log')
    # plt.savefig('fitted_diffuse.png')

    return theta_d, error_theta_d

def fit_convolution():
    theta, phi = np.loadtxt('data_convolution.dat').T

    # plt.figure()
    # plt.hist(theta, range=(0, 10), bins=50)
    # plt.xlim(0, 10)
    # plt.yscale('log')
    # plt.show()

    # plt.figure()
    # plt.xlim(0, 360)
    # plt.hist(phi, range=(0, 360), bins=36)
    # plt.show()

    bins = 25
    range = (0, 10)
    hist, edges = np.histogram(theta, bins=bins, range=range)
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1:] - edges[:-1]

    c0 = 1.22 / np.pi**1.5
    sigma = 0.3
    m = Minuit(lambda N, theta_d: -2*Lnlike(Convolution(sigma), x, hist, width)((N, c0, theta_d)), N=len(theta), theta_d=5, error_N=0.1, error_theta_d=0.01, limit_N=(0, 1e7), limit_theta_d=(0, 10))
    m.migrad()
    m.hesse()
    theta_d = m.values['theta_d']
    error_theta_d = m.errors['theta_d']

    # N = m.values['N']
    # f = N * Convolution(sigma)(x, c0, theta_d) * width

    # plt.figure()
    # plt.plot(x, f)
    # plt.hist(theta, bins=bins, range=range)
    # plt.xlim(0, 10)
    # plt.yscale('log')
    # plt.savefig('fitted_convolution.png')

    return theta_d, error_theta_d


if __name__ == "__main__":
    c0 = 1.22 / np.pi**1.5
    theta_d = 5.5
    x_diffuse = np.linspace(0, 10, 100)
    fd = diffuse_model(x_diffuse, c0, theta_d)

    # plt.figure()
    # plt.plot(x_diffuse, fd)
    # plt.xlim(0, 10)
    # plt.yscale('log')
    # plt.show()

    sigma = 0.3
    x_psf = np.linspace(0, 1, 10)
    p = PSF(x_psf, sigma)
    
    # plt.figure()
    # plt.plot(x, p)
    # plt.xlim(0, 10)
    # plt.ylim(0, 12)
    # plt.show()

    convolution = convolution(x_diffuse, c0, theta_d, x_psf, sigma)

    # plt.figure()
    # plt.plot(x_diffuse, convolution)
    # plt.xlim(0, 10)
    # plt.yscale('log')
    # plt.show()

    # sample_PSF(sigma)
    fitted_sigma, error_sigma = fit_PSF()
    print('Fitting PSF result:', fitted_sigma, error_sigma)

    #sample_diffuse(c0, theta_d)
    fitted_theta_d, error_theta_d = fit_diffuse()
    print('Fitting diffuse model result:', fitted_theta_d, error_theta_d)

    # simulation()
    fitted_theta_d, error_theta_d = fit_convolution()
    print('Fitting convolution result:', fitted_theta_d, error_theta_d)