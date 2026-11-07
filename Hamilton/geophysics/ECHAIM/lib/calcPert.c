#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mathlib.h"
#include "prepInd.h"
#include "getMagCoords.h"
#include "parameterPert.h"
#include "getCoefs.h"
#include "date.h"
#include "otherIndices.h"
#include "getIndices.h"
#include "global.h"
#include "calcPert.h"
#include "maxIndices.h"
#include "parameterAE.h"
#include "sqlite3.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Routine to calculate the perturbation part of the model
//input is 3 arrays, magnetic latitude and longitude
// Julian times
//arrays are the same length l0
//output is array, size of l0
double *calcPert (double *jd, double *lat, double *lon, int l0, sqlite3 *db, sqlite3 *dbCoefs)
{
	//declare variables
	//output, magnetic coordinates, altitude
	double *output, *alt; //length = l0 
	
	//interpolated indices array
	double *f10_81ip, *aeip, *dstip, *apip; //length = l0
	
	//F10.7 flux indices, smoothed to 27 days, smoothed to 81 days, time array
	double *f10_81, *f10x; //length = lf10
	int lf10;
	
	//AE indices, integrated AE
	double *aex, *aeInt; //length of lae
	int lae;
	
	//DSt indices, integrated DST (length and time array same as ae)
	double *dstInt, *dstx; //length of lae
	int ldst;
	
	//Ap indices, integrated Ap
	double *apx, *apInt; //length of lap
	int lap;
	
	//lower and upper time bounds, solar zenith angle, dipole tilt
	double *jdLower, *jdUpper, *dpLower, *dpUpper; //length =l0
	
	//Model parameters output pointer
	double **paramsLower, **paramsUpper; //size of 360xl0
	
	double **coefs; //the model coefficients, size of 360x24
	
	double jd0,jd1; //min and max jd
	int r, *lx; //integer for min/max output, length placeholder
	
	//allocate memory
	alt = calloc(l0, sizeof(double));
	jdLower = calloc(l0, sizeof(double));
	jdUpper = calloc(l0, sizeof(double));
	dpUpper = calloc(l0, sizeof(double));
	dpLower = calloc(l0, sizeof(double));
	output = calloc(l0, sizeof(double));
	
	
	//setting arrays
	for (int i=0; i<l0; i++)
	{
		alt[i] = 300.0; //setting altitude to 300
		
		//need time stamps before and after the times of interest
		//set to halfway between UTC hours
		jdLower[i] = jdHourly(jd[i] + (0.5/24.0)) - 0.5/24.0;//setting lower time
		jdUpper[i] = jdHourly(jd[i] + (0.5/24.0)) + 0.5/24.0;//setting upper time
	}
	
	//get the min and max JD inputted
	//The time range will go from min(jd) - 90 days to max(jd) + 90
	//***NEED TO DECIDE ON THE MINIMUM HERE AND IMPLEMENT IT IN ALL CODE
	r = minInd(jd, l0);
	jd0 = jd[r];
	r = maxInd(jd,l0);
	jd1 = jd[r];
	
	//get the F10.7 data
	f10_81 = F10_81(jd0-43.0, jd1+43.0, &f10x, &lx, db);
	lf10 = lx[0];
	
	//Get the DST, AP data
	//Time integrated
	dstInt = DST(jd0-1, jd1+1, &dstx, &lx, db);
	ldst = lx[0];
	apInt = AP(jd0-1, jd1+1, &apx, &lx, db);
	lap = lx[0];

	//interpolate the indices
	f10_81ip = interp(f10_81, f10x, lf10, jd, l0);
	dstip = interp(dstInt, dstx, ldst, jd, l0);
	apip = interp(apInt, apx, lap, jd, l0);
	
	//Check if AP or DST is outside the range of available data
	if (jd1+1 > maxDST(1,NULL))
	{
		double mdst = maxDST(1,NULL);
		
		for (int i=0; i<l0; i++)
		{
			if (jd[i] > mdst) dstip[i] = -10.215;
		}
	}
	
	if (jd1+1 > maxAP(1,NULL))
	{
		double map = maxAP(1,NULL);
		
		for (int i=0; i<l0; i++)
		{
			if (jd[i] > map) apip[i] = 9.106;
		}
	}
	
	//get AE and interpolate
	aeInt = AE(jd0-1, jd1+1, &aex, &lx, db);
	lae = lx[0];
	
	aeip = interp(aeInt, aex, lae, jd, l0);
	
	free(aeInt);
	free(aex);
	
	//check for nans and create synthetic AE
	int pcCheck = 0;
	double *pcx, *pcInt, *pcip, **aeparams, *aecoefs, s, aepcip, *dp;
	for (int i=0; i<l0; i++)
	{
		if (aeip[i] != aeip[i])
		{
			if (!pcCheck)
			{
				pcCheck = 1;
				//get PC for the range
				pcInt = PC(jd0-1, jd1+1, &pcx, &lx, db);

				//interpolate to the input times
				pcip = interp(pcInt, pcx, lx[0], jd, l0);
				
				//dipole tilt
				dp = calloc(l0, sizeof(double));
				for (int j=0; j<l0; j++)
				{
					dp[j] = dpTilt(jd[j]);
				}
				
				aeparams = parametersAE(jd, f10_81ip, dp, pcip, l0);
				aecoefs = getAE(dbCoefs);
			}
			
			//calculate synthetic AE
			s = 0;
			aepcip = 0;
			
			for (int k=0; k<6; k++)
			{
				s += aeparams[i][k] * aecoefs[k+1];
			}
			
			aepcip = s + aecoefs[0];
			if (aepcip >= 15.0)  aeip[i] = aepcip; else aeip[i] = 15.0;

		}
	}
	
	if (pcCheck)
	{
		free(pcInt);
		free(pcx);
		for (int i=0; i<l0; i++) {free(aeparams[i]);}
		free(aeparams);
		free(aecoefs);
		free(pcip);
		free(dp);
	}
	
	//free arrays
	free(dstx);
	free(apx);
	free(f10x);
	free(alt);
	free(apInt);
	free(dstInt);
	free(f10_81);
	
	//get the solar zenith angle and the dipole tilt for the bottom and top times
	for (int i=0; i<l0; i++)
	{
		dpLower[i] = dpTilt(jdLower[i]);
		dpUpper[i] = dpTilt(jdUpper[i]);
		
	}
	
	//get model params
	paramsLower = parametersPert(lon, lat, f10_81ip, dpLower, aeip, dstip, apip, l0);
	paramsUpper = parametersPert(lon, lat, f10_81ip, dpUpper, aeip, dstip, apip, l0);
	
	//free arrays
	free(f10_81ip);
	free(dpLower);
	free(dpUpper);
	free(aeip);
	free(dstip);
	free(apip);
	free(jdLower);
	free(jdUpper);
	
	//get the model coefs
	coefs = getPERT(dbCoefs);
	
	//calc perturbation
	//looping through the inputted times
	for (int i=0; i<l0; i++)
	{
		//get the hour for the current time
		double h = (jd[i] - jdDaily(jd[i])) * 24.0 + 1e-8;
		
		//create the lower and upper time index (eg if time {0.5:1.49] then lower index zero
		//corresponding to hour 0.5
		//if less than 0/greater than 23 then wrap around
		int til = ((floor(h + 0.5) - 1) < 0) ? 23 : floor(h + 0.5) - 1;
		int tiu = (til + 1 > 23) ? 0 : til + 1;
		
		//calculate the weight based on the times
		//ti + 0.5 = time corresponding to lower time
		double wl = fabs(1.0 - (h - (floor(h - 0.5) + 0.5)));
		double wu = 1.0 - wl;
		
		double sL = 0, sU = 0; //sum variables
		
		//iterate through all the parameters returned from parameters routines
		//Calculating the sums, NmF2
		for (int j=0; j<360; j++)
		{
			sL += paramsLower[i][j] * coefs[til][j+1];
			sU += paramsUpper[i][j] * coefs[tiu][j+1];
		}
		
		output[i] = (((sL + coefs[til][0]) * wl) + ((sU + coefs[tiu][0]) * wu)) / (wl + wu); //Perturbation calculation
		
		//reset the sums to 0
		sL = 0;
		sU = 0;
		
		
		if (lat[i] < 45)
		{
			output[i] = 0.0/0.0;

		}
		
	}
	
	//free memory
	for (int i=0; i<24; i++) {free(coefs[i]);}
	free(coefs);
	
	for (int i=0; i<l0; i++)
	{
		free(paramsLower[i]);
		free(paramsUpper[i]);
	}
	free(paramsLower);
	free(paramsUpper);
	
	
	//return result
	return output;
	
}