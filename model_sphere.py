from ROOT import TFile, TH1D, TH2D, TF1, TCanvas, TStyle, TTree, TMath, TGraph, TGraphErrors, TMinuit

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def angdiff_sphere(theta1, phi1, theta2, phi2):
    theta1 = np.deg2rad(theta1)
    phi1 = np.deg2rad(phi1)
    theta2 = np.deg2rad(theta2)
    phi2 = np.deg2rad(phi2)

    z1 = np.cos(theta1)
    x1 = np.sin(theta1) * np.cos(phi1)
    y1 = np.sin(theta1) * np.cos(phi1)

    z2 = np.cos(theta2)
    x2 = np.sin(theta2) * np.cos(phi2)
    y2 = np.sin(theta2) * np.cos(phi2)

    ang = x1*x2 + y1*y2 + z1*z2
    ang = np.rad2deg(np.arccos(ang))
    return ang

def Diffuse(x, theta_d):
    c0 = 1.22 / (np.pi**1.5)
    diffuse = c0 / theta_d / (x + theta_d*0.06) * np.exp(-x**2/theta_d**2)
    return diffuse

def DHfunc_2D(theta, size, theta_d):
    xv = theta
    range_ = 1.5
    xlow = xv - range_
    xupp = xv + range_
    ylow = 0
    yupp = 360
    nstep = 100

    step = (xupp - xlow) / nstep

    for i in range(nstep):
        xx = xlow + (i-0.5) * step
        if xx < 0:
            continue
        for j in range(360):
            yy = ylow + (j-0.5)*yuu
            / 360
            dist0 = abs(xx)
            fland = Diffuse(dist0, theta_d)
            dist = angdiff_sphere(xx, yy, xv, 0)
            if dist > 1:
                continue
            sum_ += fland * 