# Simulation and fitting the convolution of diffuse model with PSF

## diffuse model

spatial diffuse model as a function of of angle $\theta$:
$$\frac{dN}{d\Omega} = N_0 \frac{1.22}{\pi^{3/2}\theta_d (\theta + 0.06\theta_d)} e^{-\frac{\theta^2}{\theta_d^2}}$$

## PSF

2D gaussian distribution with $\sigma$:
$$\frac{dN}{\Omega} = \frac{1}{2\pi\sigma^2}e^{-\frac{\theta^2}{2\sigma^2}}$$

Note:

From 1D gaussian distribution with $\sigma$: 
$$f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{\sigma^2}}$$ 
To 2D gaussian distribution:
$$\int \int f(x, y) dxdy 
= \int \int \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{\sigma^2}} \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{y^2}{\sigma^2}} dxdy
= \int \int \frac{1}{2\pi \sigma^2} e^{-\frac{x^2+y^2}{\sigma^2}} dxdy
= \int \int \frac{1}{2\pi \sigma^2} e^{-\frac{\theta^2}{\sigma^2}} \sin \theta d\theta d\phi$$

## simulation

- reject sampling method
- sample diffuse model
- sample PSF
- simulation

Note: coordinate transformation from PSF to diffuse model

## fitting the convolution

- likelihood fitting