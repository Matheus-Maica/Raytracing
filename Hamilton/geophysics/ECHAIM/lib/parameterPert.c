#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "date.h"
#include "mathlib.h"
#include "parameterPert.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

#define M_PI 3.14159265358979323846
#define D2R 0.017453292519943295769


//Calculate the model parameters for the perturbation model
//mlon = magnetic longitude
//mlat = magnetic latitude
//f10_81 = indices smoothed to 81 days
//diptilt = dipole tilt angle
//ae = AE indices
//dst = DST indices
//ap = Ap indices
//l0 is the length of the arrays (they are all the same length)
double ** parametersPert (double *mlon, double *mlat, double *f10_81, double *dipTilt, \
							double *ae, double *dst, double *ap, int l0)
{
	//define variables
	double *cosFactor, *sinTilt, *cosTilt, *f0, *f1, *f2, *f3;
	double *smLon, *cmLon;
	int c=0;
	static double **output;
	
	//allocate memory
	cosFactor = calloc(l0, sizeof(double));
	sinTilt = calloc(l0, sizeof(double));
	cosTilt = calloc(l0, sizeof(double));
	f0 = calloc(l0, sizeof(double));
	f1 = calloc(l0, sizeof(double));
	f2 = calloc(l0, sizeof(double));
	f3 = calloc(l0, sizeof(double));
	smLon = calloc(l0, sizeof(double));
	cmLon = calloc(l0, sizeof(double));
	
	output = (double **) calloc(l0, sizeof(double));

	//this finishes allocating the 2D output array output[183,l0]
	for (int i=0; i<l0; i++)
	{
		output[i] = (double *) calloc(360, sizeof(double));
	}
	
	//set the arrays
	for (int i=0; i<l0; i++)
	{
		cosFactor[i] = cos((90.0 - mlat[i]) * (M_PI / 45.0));
		sinTilt[i] = sin(dipTilt[i]);
		cosTilt[i] = cos(dipTilt[i]);
		f0[i] = sqrt(f10_81[i]);
		f1[i] = exp((-1.0) * (ap[i] / 30.0));
		f2[i] = exp(ae[i] / 700.0);
		f3[i] = exp(dst[i] / 300.0);
	}
	
	for (int i=0; i<6; i++)
	{
		//if i < 4 then iterate to i, else iterate to 4
		for (int j=0; j<=((i < 3) ? i : 3); j++)
		{
			//set the smLon and cmLon arrays
			for (int ai=0; ai<l0; ai++)
			{
				smLon[ai] = sin(j * D2R * mlon[ai]);
				cmLon[ai] = cos(j * D2R * mlon[ai]);
			}
			
			if (j == 0)
			{
				for (int ai=0; ai<l0; ai++)
				{
					double le = legendre(cosFactor[ai], i, 0);
					
					output[ai][c] = f1[ai] * sinTilt[ai] * f0[ai] * le;
					output[ai][c+1] = f1[ai] * sinTilt[ai] * le;
					output[ai][c+2] = f2[ai] * sinTilt[ai] * f0[ai] * le;
					output[ai][c+3] = f2[ai] * sinTilt[ai] * le;
					
					output[ai][c+4] = f1[ai] * cosTilt[ai] * f0[ai] * le;
					output[ai][c+5] = f1[ai] * cosTilt[ai] * le;
					output[ai][c+6] = f2[ai] * cosTilt[ai] * f0[ai] * le;
					output[ai][c+7] = f2[ai] * cosTilt[ai] * le;
					
					output[ai][c+8] = f3[ai] * sinTilt[ai] * f0[ai] * le;
					output[ai][c+9] = f3[ai] * sinTilt[ai] * le;
					output[ai][c+10] = f3[ai] * cosTilt[ai] * f0[ai] * le;
					output[ai][c+11] = f3[ai] * cosTilt[ai] * le;
				}
				
				c+=12;
				
			}
			else
			{
				for (int ai=0; ai<l0; ai++)
				{
					double le = legendre(cosFactor[ai], i, j);
					
					output[ai][c] = smLon[ai] * sinTilt[ai] * f1[ai] * f0[ai] * le;
					output[ai][c+1] = cmLon[ai] * sinTilt[ai] * f1[ai] * f0[ai] * le;
					output[ai][c+2] = smLon[ai] * sinTilt[ai] * f1[ai] * le;
					output[ai][c+3] = cmLon[ai] * sinTilt[ai] * f1[ai] * le;
					
					output[ai][c+4] = smLon[ai] * sinTilt[ai] * f2[ai] * f0[ai] * le;
					output[ai][c+5] = cmLon[ai] * sinTilt[ai] * f2[ai] * f0[ai] * le;
					output[ai][c+6] = smLon[ai] * sinTilt[ai] * f2[ai] * le;
					output[ai][c+7] = cmLon[ai] * sinTilt[ai] * f2[ai] * le;
					
					output[ai][c+8] = smLon[ai] * cosTilt[ai] * f1[ai] * f0[ai] * le;
					output[ai][c+9] = cmLon[ai] * cosTilt[ai] * f1[ai] * f0[ai] * le;
					output[ai][c+10] = smLon[ai] * cosTilt[ai] * f1[ai] * le;
					output[ai][c+11] = cmLon[ai] * cosTilt[ai] * f1[ai] * le;
					
					output[ai][c+12] = smLon[ai] * cosTilt[ai] * f2[ai] * f0[ai] * le;
					output[ai][c+13] = cmLon[ai] * cosTilt[ai] * f2[ai] * f0[ai] * le;
					output[ai][c+14] = smLon[ai] * cosTilt[ai] * f2[ai] * le;
					output[ai][c+15] = cmLon[ai] * cosTilt[ai] * f2[ai] * le;
					
					output[ai][c+16] = smLon[ai] * sinTilt[ai] * f3[ai] * f0[ai] * le;
					output[ai][c+17] = cmLon[ai] * sinTilt[ai] * f3[ai] * f0[ai] * le;
					output[ai][c+18] = smLon[ai] * sinTilt[ai] * f3[ai] * le;
					output[ai][c+19] = cmLon[ai] * sinTilt[ai] * f3[ai] * le;
					
					output[ai][c+20] = smLon[ai] * cosTilt[ai] * f3[ai] * f0[ai] * le;
					output[ai][c+21] = cmLon[ai] * cosTilt[ai] * f3[ai] * f0[ai] * le;
					output[ai][c+22] = smLon[ai] * cosTilt[ai] * f3[ai] * le;
					output[ai][c+23] = cmLon[ai] * cosTilt[ai] * f3[ai] * le;
				}
				
				c+=24;
			}
		}
	}
	
	//clear the memory
	free(cosFactor);
	free(sinTilt);
	free(cosTilt);
	free(smLon);
	free(cmLon);
	free(f0);
	free(f1);
	free(f2);
	free(f3);
	
	return output;
	
}       