import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data2.txt')
print(data.shape)
psf_theta = data[:, 0]
psf_phi = data[:, 1]
diffuse_theta = data[:, 2]
diffuse_phi = data[:, 3]
dTheta = data[:, 4]

psf_theta2, psf_phi2 = np.loadtxt('data/data_PSF.dat')[:200000].T
print(psf_theta2.shape)
diffuse_theta2, diffuse_phi2 = np.loadtxt('data/data_diffuse.dat')[:200000].T
print(diffuse_theta2.shape)
convolution_theta2, convolution_phi2 = np.loadtxt('data/data_convolution.dat').T
print(convolution_theta2.shape)

plt.figure()
plt.hist(psf_theta, bins=100, density=True, label='gyy')
plt.hist(psf_theta2, bins=100, density=True, label='dkk')
plt.xlabel('Gaus_theta')
plt.legend()
plt.savefig('/mnt/c/Users/DKKYJ/Desktop/Gaus_theta.png')

plt.figure()
plt.hist(diffuse_theta, bins=100, density=True, label='gyy')
plt.hist(diffuse_theta2, bins=100, density=True, label='dkk')
plt.xlabel('model_theta')
plt.legend()
plt.savefig('/mnt/c/Users/DKKYJ/Desktop/model_theta.png')

plt.figure()
plt.hist(dTheta, bins=100, density=True, label='gyy')
plt.hist(convolution_theta2, bins=100, density=True, label='dkk')
plt.xlabel('angular_difference')
plt.legend()
plt.savefig('/mnt/c/Users/DKKYJ/Desktop/angular_difference.png')
