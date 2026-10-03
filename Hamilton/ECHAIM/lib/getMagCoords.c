#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "aacgmlib_v2.h"
#include "mlt_v2.h"
#include "date.h"
#include "getMagCoords.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************


//routine to get the magnetic latitude, longitude and local time
//expects a julian time as input, location in geographic coordinates
//height in km
//output array with indices:
//[mlat, mlon, mlt]
double * getMagCoords(double jd, double lat, double lon, double hgt, int mltCheck)
{
	//set variables
	static double *output; //return array
	double x; //unused but necessary variable for AACGM convert routine
	int err, *r; //aacgm error return, array for greg date
	
	//allocate array memory
	output = calloc(3, sizeof(double)); 
	
	//get the gregdate for the input time
	r = gregDate(jd);

	//set time for AACGM routines
	AACGM_v2_SetDateTime(r[0],r[1],r[2],r[3],r[4],r[5]);
		
	//get mlat and mlon
	err = AACGM_v2_Convert(lat, lon, hgt, &output[0], &output[1], &x, G2A);
	if (err) printf("AACGM_v2_Convert Error\n");

	if (mltCheck != 0)
	{
		//need to unset DateTime before MLT call
		AACGM_v2_SetDateTime(1990,1,1,0,0,0);

		//get mlt
		output[2] = MLTConvertYMDHMS_v2(r[0],r[1],r[2],r[3],r[4],r[5],output[1]);
	} else
	{
		output[2] = NAN;
	}
	
	free(r);
	
	return output;
	
}
