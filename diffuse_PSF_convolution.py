import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.signal import convolve, fftconvolve

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

def convolution(x, c0, theta_d, sigma):
    model = diffuse_model(x, c0, theta_d)
    psf = PSF(x, sigma)
    convolution = fftconvolve(model, psf, mode='same')
    return convolution

def sample_phi(Nsample):
    phi = np.random.uniform(0, 360, Nsample)
    return phi

def sample_diffuse(c0, theta_d):
    N = 100000
    x = np.random.uniform(0, 6, N)
    y = diffuse_model(x, c0, theta_d)
    u = np.random.uniform(0, 1.2, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data_source.dat', np.array([theta, phi]).T)
    return theta, phi

def sample_PSF(sigma):
    N = 1000
    x = np.random.uniform(0, 1, N)
    y = PSF(x, sigma)
    u = np.random.uniform(0, 12, N)
    theta = x[u<y]
    phi = sample_phi(len(theta))
    np.savetxt('data_PSF.dat', np.array([theta, phi]).T)
    return theta, phi

def transformation(theta_s, phi_s, theta_psf, phi_psf):
    theta_psf = np.deg2rad(theta_psf)
    phi_psf = np.deg2rad(phi_psf)
    theta_s = np.deg2rad(theta_s)
    phi_s = np.deg2rad(phi_s)

    Rz = np.array([[np.cos(phi_s), -np.sin(phi_s), 0], [np.sin(phi_s), np.cos(phi_s), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(theta_s), 0, -np.sin(theta_s)], [0, 1, 0], [np.sin(theta_s), 0, np.cos(theta_s)]])
    vector = np.array([np.sin(theta_psf)*np.cos(phi_psf), np.sin(theta_psf)*np.sin(phi_psf), np.cos(theta_psf)]).T
    x, y, z = np.dot(Rz, np.dot(Ry, vector))
    assert abs(x**2+y**2+z**2-1)<1e-9
    theta = np.rad2deg(np.arccos(z))
    phi = np.rad2deg(np.arctan2(y, x))
    return theta, phi

def simulation(c0, theta_d, sigma):
    theta_S, phi_S = sample_diffuse(c0, theta_d)
    theta_PSF, phi_PSF = sample_PSF(sigma)

    theta_PH = []
    phi_PH = []
    for theta_s, phi_s in zip(theta_S, phi_S):
        for theta_psf, phi_psf in zip(theta_PSF, phi_PSF):
            theta, phi = transformation(theta_s, phi_s, theta_psf, phi_psf)
            theta_PH.append(theta)
            phi_PH.append(phi)
    np.savetxt('data_PH.dat', np.array([theta_PH, phi_PH]).T)
    return theta_PH, phi_PH

def fit():
    pass

if __name__ == "__main__":
    c0 = 1.22 / np.pi**1.5
    theta_d = 5.5
    x = np.linspace(0, 6, 1000)
    fd = diffuse_model(x, c0, theta_d)

    # plt.figure()
    # plt.plot(x, fd)
    # plt.xlim(0, 6)
    # plt.ylim(0, 0.12)
    # plt.show()

    sigma = 0.3
    p = PSF(x, sigma)
    
    # plt.figure()
    # plt.plot(x, p)
    # plt.xlim(0, 6)
    # plt.ylim(0, 12)
    # plt.show()

    convolution = convolution(x, c0, theta_d, sigma)

    # plt.figure()
    # plt.plot(x, convolution)
    # plt.xlim(0, 6)
    # plt.ylim(0, 6)
    # plt.show()

    simulation(c0, theta_d, sigma)