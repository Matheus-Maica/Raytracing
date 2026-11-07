#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "date.h"
#include "mathlib.h"
#include "parameterHmF2.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

#define M_PI 3.14159265358979323846
#define D2R 0.017453292519943295769

//Calculate the model parameters for the HmF2 model
//mlat = magnetic latitude; mlon = magnetic" longitude
//jd = julian dat, sza = solar zenith angle
//ig = IG indices, f10_27 = f10.7 indices smoothed to 27 days
//F10.7_81 = indices smoothed to 81 days
//dt = dipole tilt angle
//l0 is the length of the arrays (they are all the same length)
double ** parametersHmF2 (double *mlon, double *mlat, double *jd, double *slz, double *ig, double *f10_27, double *f10_81, double *dipTilt, int l0)
{
	//define variables
	double *f1, *cosFactor, *doy, *cosSlz, *sinSlz, *sinTilt, *cosTilt, *s2;
	double *smLon, *cmLon, *sDOY, *cDOY;
	double *plm;
	double carr[5] = {0.2, 0.25, 1.0/3.0, 0.5, 1.0};
	static double **output;
	int c=0, *gd; //index counter for the output array, gregDate pointer
	
	//allocate the memory
	cosFactor = calloc(l0, sizeof(double));
	f1 = calloc(l0, sizeof(double));
	doy = calloc(l0, sizeof(double));
	cosSlz = calloc(l0, sizeof(double));
	sinSlz = calloc(l0, sizeof(double));
	sinTilt = calloc(l0, sizeof(double));
	cosTilt = calloc(l0, sizeof(double));
	s2 = calloc(l0, sizeof(double));
	plm = calloc(l0, sizeof(double));
	smLon = calloc(l0, sizeof(double));
	cmLon = calloc(l0, sizeof(double));
	sDOY = calloc(l0, sizeof(double));
	cDOY = calloc(l0, sizeof(double));
	output = (double **) calloc(l0, sizeof(double));
	
	//this finishes allocating the 2D output array output[183,l0]
	for (int i=0; i<l0; i++)
	{
		output[i] = (double *) calloc(183, sizeof(double));
	}

	//set the arrays
	for (int i=0; i<l0; i++)
	{
		gd = gregDate(jd[i]);
		doy[i] = jd[i] - julianDate(gd[0], 1, 1, 0, 0, 0);
		f1[i] = pow(f10_81[i],0.275);
		cosFactor[i] = cos((90.0 - mlat[i]) * (M_PI / 45.0));
		cosSlz[i] = cos(D2R * slz[i]);
		sinSlz[i] = sin(D2R * slz[i]);
		sinTilt[i] = sin(dipTilt[i]);
		cosTilt[i] = cos(dipTilt[i]);
		s2[i] = pow(sin(M_PI * (doy[i] / 365.25)), 2.0);
		
		free(gd);
	}
	
	for (int i=0; i<4; i++) //make it to i=4, j=1, k =4 
	{
		//if i < 2 then iterate to i, else iterate to 2
		for (int j=0; j<=((i < 2) ? i : 2); j++)
		{
			//set the smLon and cmLon arrays
			for (int ai=0; ai<l0; ai++)
			{
				smLon[ai] = sin(j * D2R * mlon[ai]);
				cmLon[ai] = cos(j * D2R * mlon[ai]);
			}
			
			for (int k=0; k<5; k++)
			{	
				//set the sDOY and CDOY arrays
				for (int ai=0; ai<l0; ai++)
				{
					sDOY[ai] = sin((2.0 * M_PI * doy[ai]) / (365.25 * carr[k]));
					cDOY[ai] = cos((2.0 * M_PI * doy[ai]) / (365.25 * carr[k]));
				}
				
				if (j == 0)
				{
					for (int ai=0; ai<l0; ai++)
					{
						double le = legendre(cosFactor[ai], i, 0);
						
						output[ai][c] = sDOY[ai] * f1[ai] * le;
						output[ai][c+1] = cDOY[ai] * f1[ai] * le;
					}
					
					c+=2;
				}else
				{
					for (int ai=0; ai<l0; ai++)
					{
						double le = legendre(cosFactor[ai], i, j);
						
						output[ai][c] = sDOY[ai] * smLon[ai] * f1[ai] * le;
						output[ai][c+1] = cDOY[ai] * smLon[ai] * f1[ai] * le;
						output[ai][c+2] = sDOY[ai] * cmLon[ai] * f1[ai] * le;
						output[ai][c+3] = cDOY[ai] * cmLon[ai] * f1[ai] * le;
					}

					c+=4;
				}
				
			}
			
			if (j == 0)
			{
				for (int ai=0; ai<l0; ai++)
				{
					double le = legendre(cosFactor[ai], i, 0);
					
					output[ai][c] = f1[ai] * s2[ai] * le;
					output[ai][c+1] = f1[ai] * sinTilt[ai] * le;
				}
				
				c+=2;
				
			} else
			{
				for (int ai=0; ai<l0; ai++)
				{
					double le = legendre(cosFactor[ai], i, j);
					
					output[ai][c] = smLon[ai] * s2[ai] * f1[ai] * le;
					output[ai][c+1] = cmLon[ai] * s2[ai] * f1[ai] * le;
					output[ai][c+2] = f1[ai] * sinTilt[ai] * smLon[ai] * le;
					output[ai][c+3] = f1[ai] * sinTilt[ai] * cmLon[ai] * le;
				}				
				
				c+=4;
			}
		}
	}
	
	for (int ai=0; ai<l0; ai++)
	{
		output[ai][c] = f10_27[ai] * cosSlz[ai];
		output[ai][c+1] =  f10_27[ai] * f10_27[ai] * cosSlz[ai];
		output[ai][c+2] = sqrt(f10_27[ai]) * cosSlz[ai];
		output[ai][c+3] = sqrt(f10_27[ai]) * sinSlz[ai];
		output[ai][c+4] = f10_27[ai] * sinSlz[ai];
		output[ai][c+5] = ig[ai] * sinSlz[ai];
		output[ai][c+6] = ig[ai] * cosSlz[ai];
		output[ai][c+7] = ig[ai];
		output[ai][c+8] = ig[ai] * ig[ai];
		output[ai][c+9] = sinSlz[ai];
		output[ai][c+10] = cosSlz[ai];
		output[ai][c+11] = cosSlz[ai] * cosTilt[ai];
		output[ai][c+12] = sinTilt[ai];
		output[ai][c+13] = cosTilt[ai];
		output[ai][c+14] = cosSlz[ai] * sinTilt[ai];
	}
	
	//free all the arrays before returning the final output
	free(f1);
	free(cosFactor);
	free(doy);
	free(cosSlz);
	free(sinSlz);
	free(sinTilt);
	free(cosTilt);
	free(s2);
	free(plm);
	free(smLon);
	free(cmLon);
	free(sDOY);
	free(cDOY);
	
	return output;
}