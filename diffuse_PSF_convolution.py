import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.signal import convolve, convolve2d
from scipy.interpolate import interp1d
from iminuit import Minuit

def diffuse_model(x, c0, theta_d):
    fd = 2*np.pi * np.sin(np.deg2rad(x)) * c0 / theta_d / (x + 0.06*theta_d) * np.exp(-x**2/theta_d**2)
    return fd

def diffuse_model_2d(x, y, c0, theta_d):
    theta = np.sqrt(x**2+y**2)
    fd = c0 / theta_d / (theta + 0.06*theta_d) * np.exp(-theta**2/theta_d**2)
    return fd

def PSF_2D_scipy(point, sigma):
    return multivariate_normal.pdf(point, mean=[0, 0], cov=[[sigma, 0], [0, sigma]])

def PSF_2D(x, y, sigma):
    return 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/sigma**2/2)

def PSF(x, sigma):
    # return 2*np.pi * np.sin(np.deg2rad(x)) * 1/sigma**2 * np.exp(-x**2/sigma**2/2)
    # return np.sin(np.deg2rad(x)) * 1/sigma**2 * np.exp(-x**2/sigma**2/2)
    return x * 1/sigma**2 * np.exp(-x**2/sigma**2/2)

def convolution(x_diffuse, c0, theta_d, x_psf, sigma):
    diffuse = diffuse_model(x_diffuse, c0, theta_d)
    psf = PSF(x_psf, sigma)
    convolution = convolve(diffuse, psf, mode='full')[:len(x_diffuse)]
    return convolution

def sample_phi(Nsample):
    phi = np.random.uniform(0, 360, Nsample)
    return phi

def sample_diffuse(c0, theta_d):
    N = 100000
    x = np.random.uniform(0, 10, N)
    y = diffuse_model(x, c0, theta_d)
    u = np.random.uniform(0, 0.004, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data/data_diffuse.dat', np.array([theta, phi]).T)
    return theta, phi

def sample_PSF(sigma):
    N = 100000
    x = np.random.uniform(0, 2, N)
    y = PSF(x, sigma)
    u = np.random.uniform(0, 2.2, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data/data_PSF.dat', np.array([theta, phi]).T)
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
    Theta_diffuse, Phi_diffuse = np.loadtxt('data/data_diffuse.dat')[:3000].T
    Theta_PSF, Phi_PSF = np.loadtxt('data/data_PSF.dat')[:3000].T

    theta_conv = []
    phi_conv = []
    for theta_diffuse, phi_diffuse in zip(Theta_diffuse, Phi_diffuse):
        for theta_PSF, phi_PSF in zip(Theta_PSF, Phi_PSF):
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
        diffuse = diffuse_model(x_diffuse, c0, theta_d)
        convolution = convolve(diffuse, self.psf, mode='full')[:len(x_diffuse)]
        return interp1d(x_diffuse, convolution, kind='cubic')(x)

    def __call__(self, x, c0, theta_d):
        return self.interp(x, c0, theta_d)

def fit_PSF():
    theta, phi = np.loadtxt('data/data_PSF.dat').T

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

    m = Minuit(lambda N, sigma: -2*Lnlike(PSF, x, hist, width)((N, sigma)), 
    #m = Minuit(lambda N, sigma: Chi2(PSF, x, hist, width)((N, sigma)),
    N=len(theta), sigma=0.32, error_N=0.1, error_sigma=0.01, limit_N=(0, 1e7), limit_sigma=(0, 0.5))
    m.migrad()
    m.hesse()
    sigma = m.values['sigma']
    error_sigma = m.errors['sigma']

    N = m.values['N']
    f = N * PSF(x, sigma) * width

    plt.figure()
    plt.plot(x, f)
    plt.hist(theta, bins=bins, range=range_)
    plt.xlim(0, 2)
    plt.savefig('data/fitted_PSF.png')

    return sigma, error_sigma

def fit_diffuse():
    theta, phi = np.loadtxt('data/data_diffuse.dat').T

    plt.figure()
    plt.hist(theta, range=(0, 10), bins=10)
    plt.xlim(0, 10)
    plt.savefig('data/hist_diffuse.png')

    # plt.figure()
    # plt.hist(phi, range=(0, 360), bins=36)
    # plt.xlim(0, 360)
    # plt.show()

    bins = 100
    range_ = (0, 10)
    hist, edges = np.histogram(theta, bins=bins, range=range_)
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

    N = m.values['N']
    f = N * diffuse_model(x, c0, theta_d) * width

    plt.figure()
    plt.plot(x, f)
    plt.hist(theta, bins=bins, range=range_)
    plt.xlim(0, 10)
    # plt.yscale('log')
    plt.savefig('data/fitted_diffuse.png')

    return theta_d, error_theta_d

def fit_convolution():
    theta, phi = np.loadtxt('data/data_convolution.dat').T

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
    range_ = (0, 10)
    hist, edges = np.histogram(theta, bins=bins, range=range_)
    x = (edges[:-1] + edges[1:]) / 2
    width = edges[1:] - edges[:-1]

    c0 = 1.22 / np.pi**1.5
    sigma = 0.3
    m = Minuit(lambda N, theta_d: -2*Lnlike(Convolution(sigma), x, hist, width)((N, c0, theta_d)), N=len(theta), theta_d=5, error_N=0.1, error_theta_d=0.01, limit_N=(0, 1e7), limit_theta_d=(0, 10))
    m.migrad()
    m.hesse()
    theta_d = m.values['theta_d']
    error_theta_d = m.errors['theta_d']

    N = m.values['N']
    f = N * Convolution(sigma)(x, c0, theta_d) * width

    plt.figure()
    plt.plot(x, f)
    plt.hist(theta, bins=bins, range=range_)
    plt.xlim(0, 10)
    plt.yscale('log')
    plt.savefig('data/fitted_convolution.png')

    return theta_d, error_theta_d


class Lnlike2D:
    def __init__(self, func, x, y, z, xwidth, ywidth):
        self.func = func
        self.x, self.y, self.z = x, y, z
        self.xwidth, self.ywidth = xwidth, ywidth

    def lnlike(self, Nargs):
        N, *args = Nargs
        #xx, yy = np.meshgrid(self.x, self.y)
        #points = np.array([xx, yy]).T
        self.f = N * self.func(self.x, self.y, *args) * self.xwidth * self.ywidth
        self.f[self.f < 1e-99] = 1e-99
        lnlike = sum([z*np.log(f) - f for (z, f) in zip(self.z.flatten(), self.f.flatten())])
        #print(Nargs, lnlike)
        return lnlike

    def __call__(self, Nargs):
        return self.lnlike(Nargs)


def fit_PSF_2d():
    theta, phi = np.loadtxt('data/data_PSF.dat').T
    x = theta * np.cos(np.deg2rad(phi))
    y = theta * np.sin(np.deg2rad(phi))

    xx, yy = np.loadtxt('data/data_PSF_2d.dat').T
    tt = np.sqrt(xx**2 + yy**2)
    pp = np.rad2deg(np.arctan2(yy, xx)) + 180

    plt.figure()
    plt.hist(tt, bins=100, range=(0, 2),density=True)
    plt.hist(theta, bins=100, range=(0, 2), density=True)
    plt.xlim(0, 2)
    plt.savefig('data/hist_theta.png')

    plt.figure()
    plt.hist(pp, bins=100, range=(0, 360), density=True)
    plt.hist(phi, bins=100, range=(0, 360), density=True)
    plt.xlim(0, 360)
    plt.savefig('data/hist_phi.png')

    plt.figure()
    plt.hist(x, bins=100, range=(-2, 2), density=True)
    plt.hist(xx, bins=100, range=(-2, 2), density=True)
    plt.xlim(-2, 2)
    plt.savefig('data/hist_x.png')

    plt.figure()
    plt.hist(y, bins=100, range=(-2, 2), density=True)
    plt.hist(yy, bins=100, range=(-2, 2), density=True)
    plt.xlim(-2, 2)
    plt.savefig('data/hist_y.png')

    bins = 50
    xrange_ = (-1, 1)
    yrange_ = (-1, 1)
    range_=[xrange_, yrange_]
    hist_2d, xedges, yedges = np.histogram2d(xx, yy, bins=bins, range=range_)

    # plt.figure()
    # plt.pcolor(hist_2d)
    # plt.axis('equal')
    # plt.xticks([0, bins], xrange_)
    # plt.yticks([0, bins], yrange_)
    # plt.savefig('data/hist_PSF_2d.png')
    # assert False

    x = (xedges[:-1] + xedges[1:]) / 2
    y = (yedges[:-1] + yedges[1:]) / 2
    xwidth = xedges[1:] - xedges[:-1]
    ywidth = yedges[1:] - yedges[:-1]

    m = Minuit(lambda N, sigma: -2*Lnlike2D(PSF_2D, x, y, hist_2d, xwidth, ywidth)((N, sigma)), 
    #m = Minuit(lambda N, sigma: Chi2(PSF, x, hist, width)((N, sigma)),
    N=len(theta), sigma=0.32, error_N=0.1, error_sigma=0.01, limit_N=(0, 1e7), limit_sigma=(0, 0.5))
    m.migrad()
    m.hesse()
    sigma = m.values['sigma']
    error_sigma = m.errors['sigma']

    return sigma, error_sigma

def fit_diffuse_2d():
    theta, phi = np.loadtxt('data/data_diffuse.dat').T
    x = theta * np.cos(np.deg2rad(phi))
    y = theta * np.sin(np.deg2rad(phi))

    bins = 100
    xrange_ = (-10, 10)
    yrange_ = (-10, 10)
    range_=[xrange_, yrange_]
    hist_2d, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range_)

    plt.figure()
    plt.pcolor(hist_2d)
    plt.axis('equal')
    plt.xticks([0, bins], xrange_)
    plt.yticks([0, bins], yrange_)
    plt.savefig('data/hist_diffuse_2d.png')
    # assert False
    
    x = (xedges[:-1] + xedges[1:]) / 2
    y = (yedges[:-1] + yedges[1:]) / 2
    xwidth = xedges[1:] - xedges[:-1]
    ywidth = yedges[1:] - yedges[:-1]

    c0 = 1.22 / np.pi**1.5
    m = Minuit(lambda N, theta_d: -2*Lnlike2D(diffuse_model_2d, x, y, hist_2d, xwidth, ywidth)((N, c0, theta_d)), 
    #m = Minuit(lambda N, theta_d: Chi2(diffuse_model, x, np.log10(hist), width)((N, c0, theta_d)),
    N=len(theta), theta_d=5, error_N=0.1, error_theta_d=0.01, limit_N=(0, 1e7), limit_theta_d=(0, 10))
    m.migrad()
    m.hesse()
    theta_d = m.values['theta_d']
    error_theta_d = m.errors['theta_d']

    return theta_d, error_theta_d

def fit_convolution_2d():
    pass

if __name__ == "__main__":
    sigma = 0.3
    x_psf = np.linspace(0, 2, 100)
    psf = PSF(x_psf, sigma)

    plt.figure()
    plt.plot(x_psf, psf)
    plt.xlim(0, 2)
    # plt.yscale('log')
    plt.savefig('dN_dTheta_PSF.png')

    c0 = 1.22 / np.pi**1.5
    theta_d = 5.5
    x_diffuse = np.linspace(0, 10, 100)
    diffuse = diffuse_model(x_diffuse, c0, theta_d)

    # plt.figure()
    # plt.plot(x_diffuse, diffuse)
    # plt.xlim(0, 10)
    # # plt.yscale('log')
    # plt.savefig('dN_dTheta_diffuse.png')

    convolution = convolution(x_diffuse, c0, theta_d, x_psf, sigma)

    # plt.figure()
    # plt.plot(x_diffuse, convolution)
    # plt.xlim(0, 10)
    # # plt.yscale('log')
    # plt.savefig('dN_dTheta_convolution.png')
    # assert False

    sample_PSF(sigma)
    fitted_sigma, error_sigma = fit_PSF()
    print('Fitting PSF result:', fitted_sigma, error_sigma)

    # sample_diffuse(c0, theta_d)
    # fitted_theta_d, error_theta_d = fit_diffuse()
    # print('Fitting diffuse model result:', fitted_theta_d, error_theta_d)

    # simulation()
    # fitted_theta_d, error_theta_d = fit_convolution()
    # print('Fitting convolution result:', fitted_theta_d, error_theta_d)
    # assert False

    # x_psf = np.linspace(-1, 1, 101)
    # y_psf = np.linspace(-1, 1, 101)
    # xx, yy = np.meshgrid(x_psf, y_psf)
    # print(xx.shape, yy.shape)

    # points = np.array([xx, yy]).T
    # print(points.shape)
    # psf_2d_scipy = multivariate_normal.pdf(points, cov=[[sigma**2, 0], [0, sigma**2]])
    # print(psf_2d_scipy.shape)

    x, y = multivariate_normal.rvs(mean=[0, 0], cov=[[sigma**2, 0], [0, sigma**2]], size=1000000).T
    np.savetxt('data/data_PSF_2d.dat', np.array([x, y]).T)
    theta = np.sqrt(x**2 + y**2)
    phi = np.rad2deg(np.arctan2(y, x)) + 180

    # xx = np.linspace(-2, 2, 100)
    # yy = np.linspace(-2, 2, 100)

    # plt.figure()
    # plt.plot(xx, norm.pdf(xx, scale=sigma))
    # plt.hist(x, bins=100, range=(-2, 2), density=True)
    # plt.xlim(-2, 2)
    # plt.savefig('hist_x.png')

    # plt.figure()
    # plt.plot(yy, norm.pdf(yy, scale=sigma))
    # plt.hist(y, bins=100, range=(-2, 2), density=True)
    # plt.xlim(-2, 2)
    # plt.savefig('hist_y.png')

    # tt = np.linspace(0, 2, 100)
    # plt.figure()
    # plt.plot(tt, PSF(tt, sigma))
    # plt.hist(theta, bins=100, range=(0, 2), density=True)
    # plt.xlim(0, 2)
    # plt.savefig('hist_theta.png')

    # plt.figure()
    # plt.hist(phi, bins=100, range=(0, 360), density=True)
    # plt.xlim(0, 360)
    # plt.savefig('hist_phi.png')

    # plt.figure()
    # plt.hist2d(x, y, bins=100, range=[(-2, 2), (-2, 2)], density=True)
    # plt.axis('equal')
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    # plt.savefig('hist_PSF_2d.png')

    # psf_2d = PSF_2D(xx, yy, sigma)
    # print(psf_2d.shape)
    # np.savetxt('2d_PSF.dat', psf_2d)

    # plt.figure()
    # # plt.pcolor(psf_2d)
    # # plt.contourf(psf_2d_scipy)
    # plt.contourf(psf_2d)
    # plt.xticks(range(0, 101, 50), x_psf[::50])
    # plt.yticks(range(0, 101, 50), y_psf[::50])
    # plt.axis('equal')
    # plt.savefig('2d_PSF.png')

    # plt.figure()
    # plt.plot(x_psf, psf_2d_scipy[:, 50], label='scipy_x')
    # plt.plot(y_psf, psf_2d_scipy[50, :], label='scipy_y')

    # plt.plot(x_psf, psf_2d[:, 50], label='x')
    # plt.plot(y_psf, psf_2d[50, :], label='y')
    # plt.plot(x_psf, 1/(np.sqrt(2*np.pi)*sigma)*norm.pdf(x_psf, scale=sigma), label='model')
    # plt.legend()
    # plt.show()

    # x_diffuse = np.linspace(-10, 10, 301)
    # y_diffuse = np.linspace(-10, 10, 301)
    # xx, yy = np.meshgrid(x_diffuse, y_diffuse)
    # diffuse_2d = diffuse_model_2d(xx, yy, c0, theta_d)
    # np.savetxt('2d_diffuse.dat', diffuse_2d)

    # plt.figure()
    # plt.pcolor(diffuse_2d)
    # plt.axis('equal')
    # plt.xticks(range(0, 1001, 100), x_diffuse[::100])
    # plt.yticks(range(0, 1001, 100), y_diffuse[::100])
    # plt.savefig('2d_diffuse.png')

    # convolution_2d = convolve2d(p_2d, diffuse_2d, mode='same')
    # np.savetxt('2d_convolution.dat', convolution_2d)

    # plt.figure()
    # plt.pcolor(convolution_2d)
    # plt.axis('equal')
    # plt.savefig('2d_convolution.png')

    fitted_sigma, error_sigma = fit_PSF_2d()
    print('Fitting PSF 2D result:', fitted_sigma, error_sigma)

    # fitted_theta_d, error_theta_d = fit_diffuse_2d()
    # print('Fitting diffuse model 2D result:', fitted_theta_d, error_theta_d)

    # fitted_theta_d, error_theta_d = fit_convolution_2d()
    # print('Fitting convolution 2D result:', fitted_theta_d, error_theta_d)