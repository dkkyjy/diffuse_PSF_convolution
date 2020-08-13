////yyguo 2020/07/10
#include"stdio.h"
#include"math.h"
#include <stdlib.h>
#include "TFile.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TTree.h"
#include "TMath.h"
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TMinuit.h"
#include <iostream>
#define         D2R 0.017453292519943296
#define        R2D 57.295779513082322
#define contour 1

double psf_sigma;
double angdiff_sphere(double theta1,double phi1,double theta2,double phi2){
/////sphere coordinate theta is the same as  zenith angle
 theta1*=D2R; phi1*=D2R;
 theta2*=D2R; phi2*=D2R;
 double z1=cos(theta1),x1=sin(theta1)*cos(phi1),y1=sin(theta1)*sin(phi1);
 double z2=cos(theta2),x2=sin(theta2)*cos(phi2),y2=sin(theta2)*sin(phi2);
 double tmp=x1*x2+y1*y2+z1*z2;
 tmp=acos(tmp)*R2D;
 return tmp;
}
////minuit result
double  Diffuse(double x, double theta_d){
  double c0=1.22/5.56832799683170787; ///5.57=pow(TMath::Pi(),1.5);
  double tmp= c0*1./theta_d/(x+theta_d*0.06)*exp(-x*x/theta_d/theta_d);
//  printf("Diffuse: c0 %lf x: %lf theta_d: %lf tmp: %lf\n",c0,theta_d,x,tmp);
  return tmp;
}
//TFile *fp=new TFile("../ana_rst_profile_Convolve/0408/profileMC_Geminga_diffuse_40_100TeV.root");
//TH1D *h_crab=(TH1D *)fp->Get("hang3_sr");
///TFile *fp=new TFile("ana_rst_profile_Convolve/0408/profile_40_100TeV.root");
TH1D *h_crab;
Double_t DHfunc_2D(Double_t theta,double size, double theta_d) {
  //Fit parameters:
  //par 0  is normlize
  //par 1  is theta_d
  //par 2  is Gauss sigma

  double par[2];
  par[0]=size;
  par[1]=theta_d;

//  printf("psf_sigma: %lf \n",psf_sigma);
  // Variables
  Double_t xx,yy;
  Double_t fland;
  Double_t sum = 0.0;
  Double_t xlow,xupp;
  Double_t ylow,yupp;
  Double_t step;
  Double_t i,j;
  Double_t xv=theta;
  double dist,dist0;
  double range=1.5;

	// Range of convolution integral
	xlow = xv - range;
	xupp = xv + range;
	ylow = 0;
	yupp = 360;
	Double_t np = 100.0;      // number of convolution steps

	step = (xupp-xlow) / np;

	for(i=1.0; i<=np; i++) {     ///////theta 
	  xx = xlow + (i-.5) * step;
	  if(xx<0) continue;
	  for(j=1.0; j<=360; j++) {   ////phi
		yy = ylow + (j-.5) *yupp/360;
		dist0=fabs(xx);	
		fland =Diffuse(dist0,theta_d);
		//printf("xv: %lf dist==xx: %lf %lf model: %lf theta_d: %lf ",xv,dist0,xx,fland,theta_d);
		//printf("r: xv: %lf xx: %lf dist: %lf fland: %lf " ,xv,xx,dist,fland);
		dist=angdiff_sphere(xx,yy,xv,0);
		//printf("dist_xv: %lf xx_yy: %lf %lf\n",dist,xx,yy);
		if(dist>1){
		  //     	printf("\n");
		       	continue;
		}
//		printf("r:yy: %lf dist: %lf\n",yy,dist);
		//sum += fland * TMath::Gaus(dist,0,0.3);
		sum += fland * TMath::Gaus(dist,0,psf_sigma)*sin(dist0*D2R)*yupp/360*D2R;
		//sum += fland * h_crab->Interpolate(dist)*sin(dist0*D2R);
		//printf("Interpolate：%lf %lf %lf  %e \n",xx,xv,yy,h_crab->Interpolate(dist));
	  }
	}
  return (par[0]* sum);
}
double Gaus(double x,double width){
 return TMath::Gaus(x,0,width);
// return 1/sqrt(2*3.1415926)/width*exp(-0.5*pow(x/width,2));
}
double TF_Gaus_dN(double *x, double *par){
  return par[0]*Gaus(x[0],par[1])*sin(x[0]*0.017453292519943296);///size theta_d
}

double TF_Gaus(double *x, double *par){
  return par[0]*Gaus(x[0],par[1]);
}

double TF_DHfunc_2D_dN(double *x, double *par){
  return DHfunc_2D(x[0],par[0],par[1])*sin(x[0]*0.017453292519943296);///size theta_d
}
double TF_DHfunc_2D(double *x, double *par){
  return DHfunc_2D(x[0],par[0],par[1]);///size theta_d
}

double TF_diffuse(double *x, double *par){
 return par[0]*Diffuse(x[0],par[1]);
}
double TF_diffuse_dN(double *x, double *par){
 return par[0]*Diffuse(x[0],par[1])*sin(x[0]*0.017453292519943296);
}

static  TMinuit *minuit=new TMinuit(2);
TH1D *hon, *hoff;
TH1D *hon2, *hoff2;
TH1D *hon3, *hoff3, *hcrab;
////minuit result
double msize,msizee;
double mtheta_d,mtheta_de;

static void POINT_FUNC(int &npar, double *gin, double &f,double *par, int iflag){
  double llf=0;
  double size=par[0];
  double theta_d=par[1];
  double delt,Ns;
  double on, bkg;
  double sr;
  double  Width=hon->GetBinWidth(1);
  
  for(int i=1;i<=hon->GetNbinsX();i++){
    on=hon->GetBinContent(i);
    bkg=hoff->GetBinContent(i);
	sr=hon->GetBinCenter(i);  
  //Ns=DGfunc(sr,size,theta_d,sigma);         ///use the median value, not intergal
  Ns=DHfunc_2D(sr,size,theta_d);         ///use the median value, not intergal
 // printf("Ns=%lf *",Ns);
	sr=2*3.14*sin(sr*D2R)*Width*D2R;
 //printf("sr: %lf\n",sr);
	Ns*=sr;
	llf+=on*log(1+Ns/bkg)-Ns;
	//llf+=log(TMath::Poisson(on,Ns+bkg)-log(TMath::Poisson(on,bkg)));
//	printf("on_bkg_Ns: %lf %lf %lf Ns: %lf \n",on,bkg,on-bkg,Ns);
  }
  f=-1.0*llf;
  //printf("llf: %lf %lf %lf\n",f,size,theta_d);
}
static void POINT_FUNC_MC(int &npar, double *gin, double &f,double *par, int iflag){
  double llf=0;
  double size=par[0];
  double theta_d=par[1];
  double delt,Ns;
  double on, bkg;
  double sr;
  double  Width=hon->GetBinWidth(1);
  
  for(int i=1;i<=hon->GetNbinsX();i++){
    on=hon->GetBinContent(i);    ///dN/Domegea
    sr=hon->GetBinCenter(i);  
  Ns=DHfunc_2D(sr,size,theta_d);         ///use the median value, not intergal
 // printf("Ns=%lf *",Ns);
	sr=2*3.14*sin(sr*D2R)*Width*D2R;
 //printf("sr: %lf\n",sr);
	Ns*=sr;
//	llf+=log(TMath::Poisson(on,Ns))-log(TMath::Poisson(on,0.0001));
	bkg=10;
	 llf+=on*log(1+Ns/bkg)-Ns;
//printf("on_bkg_Ns: %lf %lf %lf Ns: %lf \n",on,bkg,on-bkg,Ns);
  }
  f=-1.0*llf;
  ///printf("llf: %lf size:  %lf theta_d: %lf\n",f,size,theta_d);
}


TGraph* point_likelihood( ){
//void  point_likelihood( ){

  minuit->mncler();
  int _flag=0;
  double arglist[10];
  arglist[0]=-1;
 minuit->mnexcm("SET PRINT",arglist,1,_flag);
  minuit->mnexcm("SET NOWarnings",arglist,1,_flag);
  minuit->mnparm(0,"size",0.01,0.001,0,0,_flag);/// start step range
  minuit->mnparm(1,"theta_d",6,0.1,0,30,_flag);   ////This need to be very precise
//  minuit->SetFCN(POINT_FUNC_MC);
  minuit->SetFCN(POINT_FUNC);

  arglist[0]=1;
  minuit->mnexcm("CALL FCN",arglist,1,_flag);
  arglist[0]=0;
  arglist[1]=0.0001;
  minuit->mnexcm("SIMPLEX",arglist,0,_flag);
  minuit->mnexcm("MIGRAD",arglist,0,_flag);
  printf("Minuit rst:");
  //minuit->mnexcm("MINOS",arglist,0,_flag);
  minuit->mnscan();

  minuit->GetParameter(0,msize,msizee);
  minuit->GetParameter(1,mtheta_d,mtheta_de);
  // cout<<"point source: "<<endl;
  printf("size: %.3e+-%.5e\t",msize,msizee);
  printf("R_d: %.2lf+-%.5lf\t",mtheta_d,mtheta_de);
  //printf("theta_d: %.3lf+-%.30lf\t",mtheta_d,mtheta_de);

  Double_t amin,edm,errdef;
  Int_t nvpar,nparx,icstat;
  minuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
  ////not gMinuit
  //cout<<" check: "<<amin<<" "<<edm<<" "<<errdef<<" "<<nvpar<<" "<<nvpar<<" "<<icstat<<endl;
  printf("sig: %.1lf\n",sqrt(-2.0*amin));
   TGraph *gr2 =NULL;
  if(contour) gr2= (TGraph*)minuit->Contour(60,0,1);
   return gr2;

  }
