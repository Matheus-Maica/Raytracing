#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "prepInd.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Do the time integration of the input
//inputs are indices, length of indices array and attenuation multiplier
//These are dependent on the index being integrated
//DST/AE = 0.95, AP = 0.75
//the JD returned will be n:l0
double *timeIntegrate(double *input, int l0, double tau, int n)
{
	//the sum, also the final value
	double *output;
	
	output = calloc(l0-n, sizeof(double));

	for (int i=0; i<l0-n; i++)
	{
		for (int j=i; j<=i+n; j++)
		{
			output[i] += (1.0 - tau) * (input[j] * (pow(tau,(double)(n-(j-i)))));	
		}
	}
	
	return output;
}

//routines to remove outlines based on provided thresholds
//input expects double array, length of array
//lower threshold, upper threshold
int outliers (double *input, int l0, double lower, double upper)
{
	int r=0;
	
	//cycle through the input array
	for (int i=0; i<l0;i++)
	{
		if (input[i] < lower || input[i] > upper) 
		{
			input[i] = NAN;
			r = 1;
		}
	}
	
	return r;
}