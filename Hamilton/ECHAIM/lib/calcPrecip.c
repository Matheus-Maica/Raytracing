#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calcPrecip.h"
#include "sqlite3.h"
#include "getIndices.h"
#include "date.h"
#include "mathlib.h"
#include "prepInd.h"
#include "runNRLMSISE.h"
#include "energyFunctions.h"
#include "getMagCoords.h"
#include "energyFluxModel.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Calculate the precipation model perturbation values
//Inptuts:
//Julian dates
//Geographic latitude
//Geographic Longitude
//Preceding inputs same length l0
//altitude array
//length of altitude array l1
//sqlite obeject for index DB
//Returns an array of perturbation values of length [l0,l1]
double **calcPrecip (double *jd, double *lat, double *lon, int l0, double *alt, int l1, sqlite3 *db)
{
	//declare variables
	double **output, **enFl; //output array, modelled energy and flux
	int year, sec, *date;
	double doy, iloc, *msisOutput, **ener_flux, *imono_rate, hr; 
	double qtot=0.0, alpha, alphaCeil, sigmoid, alpha2;
	double *mlat, *mlt; //inputs to calcs
	//indices from the DB and time
	double *f10, *f10_81, *f10x, *f1081x, *apx, *ap_8, *ae, *aex, *aeip, *pc, *pcx, *pcip;
	//inteprolated indices
	double *f10ip, *f10_81ip, *apip;
	double jd0,jd1; //min and max jd
	int r, *lx, lf10, lap, lae, lpc; //integer for min/max output, length placeholder
	//modelled energy and flux
	double *ener_mean, *eflux_mean;
	int en0 = 90; //length of output from energy spectrum (second dimension)
	
	//allocate memory for output array
	output = calloc(l0, sizeof(double));
	for (int i=0; i<l0; i++) {output[i] = calloc(l1, sizeof(double));}
	mlat = calloc(l0, sizeof(double));
	mlt = calloc(l0, sizeof(double));
	ener_mean = calloc(l0, sizeof(double));
	eflux_mean = calloc(l0, sizeof(double));
	
	//set values
	double erg2ev = 6.2415e11; //convert erg to ev
	
	//calculate magnetic coordinates
	for (int i=0; i<l0; i++)
	{	
		//get mag coords
		double *m;
		
		m = getMagCoords(jd[i], lat[i], lon[i], 300.0, 1);
		mlat[i] = m[0];
		mlt[i] = m[2];
				
		free(m);
	}
	
	//get min and max times in the array
	r = minInd(jd, l0);
	jd0 = jd[r];
	r = maxInd(jd,l0);
	jd1 = jd[r];
	
	//get the F10.7 data
	f10 = F10(jd0-43.0, jd1+43.0, &f10x, &lx, db);
	lf10 = lx[0];
	f10_81 = F10_81(jd0-43.0, jd1+43.0, &f1081x, &lx, db);
	
	//get AP data
	ap_8 = AP_8(jd0-1.0, jd1+1.0, &apx, &lx, db);
	lap = lx[0];

	//get AE data
	ae = AEraw(jd0-1, jd1+1, &aex, &lx, db);
	lae = lx[0];
	
	//get PC data
	pc = PCraw(jd0-1, jd1+1, &pcx, &lx, db);
	lpc = lx[0];
	
	//interpolate the indices
	f10ip = interp(f10, f10x, lf10, jd, l0);
	f10_81ip = interp(f10_81, f1081x, lf10, jd, l0);
	apip = interp(ap_8, apx, lap, jd, l0);
	aeip = interp(ae, aex, lae, jd, l0);
	pcip = interp(pc, pcx, lpc, jd, l0);
	
	//get the modelled energy and flux
	enFl = energyFluxModel(mlt, mlat, aeip, pcip, l0);
	
	//set energy and flux values
	for (int i=0; i<l0; i++)
	{
		if (enFl[0][i] < 0) {enFl[0][i] = 0.0;}
		if (enFl[1][i] < 0) {enFl[1][i] = 0.0;}
		ener_mean[i] = enFl[0][i] / 2.0;
		eflux_mean[i] = enFl[1][i] * erg2ev / 1000.0;
	}
	
	//loop through input arrays, running NRLMISISE and calculating precip perturbation output
	for (int i=0; i<l0; i++)
	{	

		//if the e flux is less than 0.05 then set output to 0
		//don't waste cycles on NRLMSISE
		if (eflux_mean[i] / erg2ev * 1000.0 < 0.05)
		{
			for (int j=0; j<l1; j++){output[i][j] = 0.0;}
			continue;
		}

		date = gregDate(jd[i]);
		year = date[0];
		hr = date[3] + date[4]/60.0 + date[5]/3600.0;
		sec = date[5] + date[4]*60.0 + date[3]*3600.0;
		
		doy = jdDOY(jd[i]);
		
		iloc = (lon[i] * 24.0 / 360.0) + hr;
		
		//energy_spectrum function
		
		//loop through altitude
		for (int j=0; j<l1; j++)
		{
			if (alt[j] < 400.0)
			{
				msisOutput = runNRLMSISE(year, doy, sec, lat[i], lon[i], alt[j], iloc, f10ip[i], f10_81ip[i], apip[i]);
			}
			else //if alt is greater than 400 km then set msisOutput values to NaN
			{
				msisOutput = calloc(13, sizeof(double));
				for (int k=0; k<13; k++) {msisOutput[k] = 0.0/0.0;}
			}
			
			ener_flux = gauss_energy_spectrum(ener_mean[i]);
			
			for (int jj=0; jj<en0; jj++)
			{ 
				ener_flux[1][jj] = ener_flux[1][jj]*eflux_mean[i];
			}
			
			imono_rate = eion_rate(lat[i], ener_flux[0], ener_flux[1], alt[j], msisOutput);
			qtot = 0.0;
			for (int jj=0;jj<en0;jj++){qtot += imono_rate[jj]*ener_flux[2][jj];}		
			alpha = 4.3e-6 * exp(-2.42e-2 * alt[j]) + 8.16e12 * exp(-0.524 * alt[j]);
			alphaCeil = (alt[j] > 130.0) ? 4.3e-6 * exp(-2.42e-2 * 130.0) + 8.16e12 * exp(-0.524*130.0) : alpha;
			
			sigmoid = 1.0 / (1.0 + exp((130.0 - alt[j]) / 40.0));
			alpha2 = alpha * (1.0 - sigmoid) + (sigmoid * alphaCeil);
			output[i][j] = qtot / alpha2;
			
			if (output[i][j] != output[i][j]) {output[i][j] = 0.0;}
			
			if (alt[j] >= 400.0) {free(msisOutput);}
		}
		
		free(date);
		
	}
	
	free(enFl[0]);
	free(enFl[1]);
	free(enFl);
	free(mlt);
	free(mlat);
	free(f10ip);
	free(f10_81ip);
	free(apip);
	free(aeip);
	free(pcip);
	free(ae);
	free(aex);
	free(f10);
	free(pc);
	free(f10x);
	free(f1081x);
	free(apx);
	free(pcx);
	free(f10_81);
	free(ap_8);
	free(ener_mean);
	free(eflux_mean);
	
	return output;
	
}