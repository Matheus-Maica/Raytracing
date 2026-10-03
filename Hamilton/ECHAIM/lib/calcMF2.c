#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "getIndices.h"
#include "mathlib.h"
#include "prepInd.h"
#include "getMagCoords.h"
#include "parameterNmF2.h"
#include "parameterHmF2.h"
#include "getCoefs.h"
#include "date.h"
#include "otherIndices.h"
#include "calcMF2.h"
#include "global.h"
#include "sqlite3.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Routine to calculate the NmF2 part of the model
//input is 3 arrays, magnetic latitude and longitude
//and Julian times
//all array are the same length l0
//option to specify only NmF2 or HmF2
//1: NmF2; 2: HmF2; 0 (anything else): both
//output is 2D double array, size of l0x2
//[0,l0] = NmF2
//[1,l0] = HmF2
double **calcMF2 (double *jd, double *glat, double *glon, double *lat, double *lon, int l0, int option, sqlite3 *db, sqlite3 *dbCoefs)
{
	//declare variables
	//output array, altitudes, magnetic coordinates
	double **output, *alt; //length = l0
	
	//interpolated indices array
	double *f10_27ip, *f10_81ip, *igip; //length = l0
	
	//F10.7 flux indices, smoothed to 27 days, smoothed to 81 days, time array
	double *f10_27, *f10_81, *f10x, *f1081x; //length = lf10
	int lf10;
	
	//lower and upper time bounds, solar zenith angle, dipole tilt
	double *jdLower, *jdUpper, *szaLower, *szaUpper, *dpLower, *dpUpper; //length =l0
	
	//Model parameters output pointer
	double **paramsLowerN, **paramsUpperN; //size of 831xl0
	double **paramsLowerH, **paramsUpperH; //size of 183xl0
	
	//IG indices
	double *ig, *igx; //length of lig
	int lig;
	
	double jd0,jd1, jd0IG, jd1IG; //min and max jd (F10 and IG), getMagCoords return pointer
	int r, *lx; //integer for min/max output, length placeholder
	
	double **coefsN, **coefsH; //the model coefficients, size of 832x24, 183x24
	
	//allocate memory
	alt = calloc(l0, sizeof(double));
	jdLower = calloc(l0, sizeof(double));
	jdUpper = calloc(l0, sizeof(double));
	szaUpper = calloc(l0, sizeof(double));
	dpUpper = calloc(l0, sizeof(double));
	szaLower = calloc(l0, sizeof(double));
	dpLower = calloc(l0, sizeof(double));
	
	//output is 2D, size of l0x2
	//[l0,0] = NmF2
	//[l0,1] = HmF2
	output = (double **)calloc(2, sizeof(double *));
	for (int i=0; i<2; i++)
	{
		output[i] = (double *) calloc(l0, sizeof(double));
	}
	
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
	//this is specific to the F10 data
	//***NEED TO DECIDE ON THE MINIMUM HERE AND IMPLEMENT IT IN ALL CODE
	r = minInd(jd, l0);
	jd0 = jd[r];
	r = maxInd(jd,l0);
	jd1 = jd[r];
	
	//setting the time range for the IG data
	//Need to be one month before and after the chosen date
	int *gd = gregDate(jd0);
	jd0IG = (gd[1] == 1) ? julianDate(gd[0]-1,12,1,0,0,0) - 1.0 : julianDate(gd[0],gd[1]-1,1,0,0,0) - 1.0;
	free(gd);
	
	gd = gregDate(jd1);
	jd1IG = (gd[1] == 12) ? julianDate(gd[0]+1,1,1,0,0,0) + 1.0 : julianDate(gd[0],gd[1]+1,1,0,0,0) + 1.0;
	free(gd);
	
	
	//get the F10.7 data
	f10_27 = F10_27(jd0-43.0, jd1+43.0, &f10x, &lx, db);
	lf10 = lx[0];
	f10_81 = F10_81(jd0-43.0, jd1+43.0, &f1081x, &lx, db);
	

	//get the IG data
	//adding 30 days to 	
	ig = IG(jd0IG, jd1IG, &igx, &lx, db);
	lig = lx[0];
	
	//interpolate the Ig and flux data to the inputted times
	f10_27ip = interp(f10_27, f10x, lf10, jd, l0);
	f10_81ip = interp(f10_81, f1081x, lf10, jd, l0);
	igip = interp(ig, igx, lig, jd, l0);

	//free the flux and IG arrays
	free(f10_27);
	free(f10_81);
	free(f10x);
	free(f1081x);
	free(igx);
	free(alt);
	free(ig);
	
	//get the solar zenith angle and the dipole tilt for the bottom and top times
	double *t; //temporary pointer for zenSun
	for (int i=0; i<l0; i++)
	{
		dpLower[i] = dpTilt(jdLower[i]);
		dpUpper[i] = dpTilt(jdUpper[i]);
		
		//input DOY, hour, latitude, longitude
		t = zenSun(jdDOY(jdLower[i]), ((jdLower[i] - jdDaily(jdLower[i])) * 24.0), glat[i], glon[i]);
		szaLower[i] = t[0];
		
		t = zenSun(jdDOY(jdUpper[i]), ((jdUpper[i] - jdDaily(jdUpper[i])) * 24.0), glat[i], glon[i]);
		szaUpper[i] = t[0];
	}

	
	//get the model parameters for the lower and upper time limit
	if (option != 2)
	{
		paramsLowerN = parametersNmF2(lon, lat, jdLower, szaLower, igip, f10_27ip, f10_81ip, dpLower, l0);
		paramsUpperN = parametersNmF2(lon, lat, jdUpper, szaUpper, igip, f10_27ip, f10_81ip, dpUpper, l0);		
	}
	
	if (option != 1)
	{
		paramsLowerH = parametersHmF2(lon, lat, jdLower, szaLower, igip, f10_27ip, f10_81ip, dpLower, l0);
		paramsUpperH = parametersHmF2(lon, lat, jdUpper, szaUpper, igip, f10_27ip, f10_81ip, dpUpper, l0);	
	}
	
	
	//free memory
	free(dpLower);
	free(dpUpper);
	free(szaLower);
	free(szaUpper);
	free(igip);
	free(f10_27ip);
	free(f10_81ip);
	free(jdLower);
	free(jdUpper);
	
	//get the model coefficients
	if (option != 2) coefsN = getNMF2(dbCoefs);
	if (option != 1) coefsH = getHMF2(dbCoefs);
	
	
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
		
		
		//double wu = fabs(1.0 - ((tiu + 0.5) - h));
		//double wl = 1.0 - wu;
		
		
		
		double sL = 0, sU = 0; //sum variables
		
		//iterate through all the parameters returned from parameters routines
		//Calculating the sums, NmF2
		if (option != 2)
		{
			for (int j=0; j<567; j++) //567
			{
				sL += paramsLowerN[i][j] * coefsN[til][j+1];
				sU += paramsUpperN[i][j] * coefsN[tiu][j+1];
			}
			
			//NmF2 for upper and lower times
			double nl = powl(10.0,(sL + coefsN[til][0]));
			double nu = powl(10.0,(sU + coefsN[tiu][0]));
			
			output[0][i] = ((nl * wl) + (nu * wu)) / (wl + wu); //NmF2
			
			//reset the sums to 0
			sL = 0;
			sU = 0;
			
			
			if (lat[i] < 45)
			{
				output[0][i] = 0.0/0.0;
				
			}
			
		}
		
		//HmF2 sums
		if (option != 1)
		{
			for (int j=0; j<183; j++)
			{
				sL += paramsLowerH[i][j] * coefsH[til][j+1];
				sU += paramsUpperH[i][j] * coefsH[tiu][j+1];
			}
			
			//HmF2 for upper and lower times
			double hl = (sL + coefsH[til][0]);
			double hu = (sU + coefsH[tiu][0]);
			
			output[1][i] = ((hl * wl) + (hu * wu)) / (wl + wu); //HmF2
			
			//reset the sums to 0
			sL = 0;
			sU = 0;
			
			
			if (lat[i] < 45)
			{
				output[1][i] = 0.0/0.0;


			}
			
		}
		
		
	}
	
	//free memory
	for (int i=0; i<24; i++)
	{
		if (option != 2) {free(coefsN[i]);}
		if (option != 1) {free(coefsH[i]);}
	}
	if (option != 2) {free(coefsN);}
	if (option != 1) {free(coefsH);}	
	
	for (int i=0; i<l0; i++)
	{
		if (option != 2)
		{
			free(paramsUpperN[i]);
			free(paramsLowerN[i]);				
		}

		if (option != 1)
		{
			free(paramsUpperH[i]);
			free(paramsLowerH[i]);				
		}	
	}
	if (option != 2)
	{
		free(paramsUpperN);
		free(paramsLowerN);				
	}

	if (option != 1)
	{
		free(paramsUpperH);
		free(paramsLowerH);				
	}	

	//return the output array
	return output;
}