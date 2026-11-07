#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "date.h"
#include "mathlib.h"
#include "parameterBot.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

#define M_PI 3.14159265358979323846
#define D2R 0.017453292519943295769

//Calculate the model parameters for the synthetic AE
//jd = julian dat, slz = solar zenith angle
//F10_81 = indices smoothed to 81 days
//diptilt = dipole tilt angle
//pc = PC indices 
//l0 is the length of the arrays (they are all the same length)
double ** parametersAE (double *jd, double *f10_81, double *dipTilt, double *pc, int l0)
{
	//define variables
	double *doy, *sinTilt, *cosTilt;
	double carr[2] = {0.5, 1.0};
	static double **output;
	int *gd; //index counter for the output array, gregDate pointer
	
	//allocate the memory
	doy = calloc(l0, sizeof(double));
	sinTilt = calloc(l0, sizeof(double));
	cosTilt = calloc(l0, sizeof(double));
	output = (double **) calloc(l0, sizeof(double));

	//this finishes allocating the 2D output array output[263,l0]
	for (int i=0; i<l0; i++)
	{
		output[i] = (double *) calloc(263, sizeof(double));
		
		//set the arrays
		gd = gregDate(jd[i]);
		doy[i] = jd[i] - julianDate(gd[0], 1, 1, 0, 0, 0);
		sinTilt[i] = sin(dipTilt[i]);
		cosTilt[i] = cos(dipTilt[i]);
		
		free(gd);
	}
	
	
	
	for (int ai=0; ai<l0; ai++)
	{
		output[ai][0] = f10_81[ai] * sin((doy[ai] * (2.0 * M_PI)) / (365.25 * carr[0]));
		output[ai][1] = f10_81[ai] * cos((doy[ai] * (2.0 * M_PI)) / (365.25 * carr[0]));
		
		output[ai][2] = f10_81[ai] * sin((doy[ai] * (2.0 * M_PI)) / (365.25 * carr[1]));
		output[ai][3] = f10_81[ai] * cos((doy[ai] * (2.0 * M_PI)) / (365.25 * carr[1]));
		
		output[ai][4] = dipTilt[ai];
		output[ai][5] = fabs(pc[ai]);
	}
	
	free(doy);
	free(sinTilt);
	free(cosTilt);
	
	return output;
	
}