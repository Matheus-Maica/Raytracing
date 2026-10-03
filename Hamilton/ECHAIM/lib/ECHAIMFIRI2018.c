#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "sqlite3.h"
#include "mathlib.h"
#include "FIRITools.h"
#include "date.h"
#include "ECHAIMFIRI2018.h"
#include "getFIRI.h"
#include "global.h"
#include "getIndices.h"
#include "otherIndices.h"

//this function calculated the D region perturbation
//inputs:
//lat = latitude
//sza = solar zenith angle
//f10 = f10.7
//doy = day of year
//l0 = length of all of the above arrays
//alt = altitudes (km)
//l1 = legnth of altitude array
double **ECHAIMFIRI2018 (double *jd, double *lat, double *lon, int l0, double *alt, int l1, sqlite3 *db)
{	
	//declare variables
	double **output; //output
	long len = 139986; //the length of the arrays
	int indices[5] = {3,14,11,3,101}; //5D length of arrays
	double tempDiff0[len], tempDiff1[len], diff[len]; //sums
	double wm=0,vm=0,xm=0,ym=0,zm=0; //max values
	double *w, *v, *x, *y, *z; //1D FIRI coordinates
	double *****w5, *****v5, *****x5, *****y5, *****z5;
	double *dPert, *****densPert;
	double minDec[5], *f10data, *f10x, *f10, jd0, jd1;
	int m = 0, r, *lx, *minI;
	//sqlite3 variables
	char cwd[1024]; //directory, sql error
	sqlite3 *dbFIRI; //SQLite3 database variable

	//setting directory
	strcpy(cwd,DIR);
	
	//get working directory and append database filename to the end
	strcat(cwd,"ECHAIM_FIRI.db");
	
	//open the DB
	int rc = sqlite3_open(cwd, &dbFIRI);
	if (rc) {printf("Error: CHAIM_DB could not be opened\n");}
	
	//allocate memory for output array
	output = calloc(l0, sizeof(double));
	for (int i=0; i<l0; i++) {output[i] = calloc(l1, sizeof(double));}
	
	//Get F10.7
	r = minInd(jd, l0);
	jd0 = jd[r];
	r = maxInd(jd,l0);
	jd1 = jd[r];
	f10data = F10(jd0-1.0, jd1+1.0, &f10x, &lx, db);
	f10 = interp(f10data, f10x, lx[0], jd, l0);
	
	//load the FIRI data from the DB file
	w = getFIRI("w",dbFIRI);
	v = getFIRI("v",dbFIRI);
	x = getFIRI("x",dbFIRI);
	y = getFIRI("y",dbFIRI);
	z = getFIRI("z",dbFIRI);
	dPert = getFIRI("denspertextra",dbFIRI);
	
	//get the 5D arrays
	w5 = arrayTo5D(w);
	v5 = arrayTo5D(v);
	x5 = arrayTo5D(x);
	y5 = arrayTo5D(y);
	z5 = arrayTo5D(z);
	densPert = arrayTo5D(dPert);
	
	//get the max values of the FIRI arrays
	m = maxInd(w,len);
	wm = abs(w[m]);
	m = maxInd(v,len);
	vm = abs(v[m]);
	m = maxInd(x,len);
	xm = abs(x[m]);
	m = maxInd(y,len);
	ym = abs(y[m]);
	m = maxInd(z,len);
	zm = abs(z[m]);
	
	//loop through array
	for (int i=0; i<l0; i++)
	{
		//check for NaNs
		if (lat[i] != lat[i] || lon[i] != lon[i]) 
		{
			continue;
		}
		
		//assign temporary values
		double tlat = lat[i];
		double tf10 = f10[i];
		double tdoy = jdDOY(jd[i]);
		double *ts = zenSun(tdoy, ((jd[i] - jdDaily(jd[i])) * 24.0), lat[i], lon[i]);
		double tsza = ts[0];
		
		//set some limits
		if (tlat > 60.0) {tlat = 60.0;}
		if (tsza > 130.0) {tsza = 130.0;}
		if (tf10 > 200.0) {tf10 = 200.0;}
		if (tf10 < 75.0) {tf10 = 75.0;}
		
		//get the diff arrays for all inputs except alt
		for (int j=0; j<len; j++)
		{
			tempDiff0[j] = abs(w[j] - tf10) / wm;
			tempDiff0[j] += abs(v[j] - tdoy) / vm;
			tempDiff0[j] += abs(x[j] - tsza) / xm;
			tempDiff0[j] += abs(y[j] - tlat) / ym;		
		}
		
		//loop through altitudes
		for (int j=0; j<l1; j++)
		{
			if (alt[j] >= 150.0 || alt[j] <= 50.0) 
			{
				output[i][j] = 0.0;
				continue;
			}
			
			// calculate the global minimum
			for (int k=0; k<len; k++)
			{
				tempDiff1[k] = tempDiff0[k] + fabs(z[k] - alt[j]) / zm;	
			}
			
			m = minInd(tempDiff1,len); //get the minimum 1D index
			minI = indexTo5D(m);
			
			//calculate the decimal indices for interpolation
			//F10.7
			if (w[m] == tf10)
			{
				minDec[0] = 0;
			} else if (tf10 > w[m])
			{
				double val0 = w5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				double val1 = w5[minI[0]+1][minI[1]][minI[2]][minI[3]][minI[4]];
				minDec[0] = ((tf10 - val0) / (val1 - val0));
			} else
			{
				double val0 = w5[minI[0]-1][minI[1]][minI[2]][minI[3]][minI[4]];
				double val1 = w5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				
				minI[0]--;
				minDec[0] = ((tf10 - val0) / (val1 - val0));
			}
			//DOY
			if (v[m] == tdoy)
			{
				minDec[1] = 0;
			} else if (tdoy > v[m])
			{
				double val0 = v5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				double val1 = v5[minI[0]][minI[1]+1][minI[2]][minI[3]][minI[4]];
				minDec[1] = ((tdoy - val0) / (val1 - val0));
			} else
			{
				double val0 = v5[minI[0]][minI[1]-1][minI[2]][minI[3]][minI[4]];
				double val1 = v5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				
				minI[1]--;
				minDec[1] = ((tdoy - val0) / (val1 - val0));
			}
			//SZA
			if (x[m] == tsza)
			{
				minDec[2] = 0;
			} else if (tsza > x[m])
			{
				double val0 = x5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				double val1 = x5[minI[0]][minI[1]][minI[2]+1][minI[3]][minI[4]];
				minDec[2] = ((tsza - val0) / (val1 - val0));
			} else
			{
				double val0 = x5[minI[0]][minI[1]][minI[2]-1][minI[3]][minI[4]];
				double val1 = x5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				
				minI[2]--;
				minDec[2] = ((tsza - val0) / (val1 - val0));
			}
			//LAT
			if (y[m] == tlat)
			{
				minDec[3] = 0;
			} else if (tlat > y[m])
			{
				double val0 = y5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				double val1 = y5[minI[0]][minI[1]][minI[2]][minI[3]+1][minI[4]];
				minDec[3] = ((tlat - val0) / (val1 - val0));
			} else
			{
				double val0 = y5[minI[0]][minI[1]][minI[2]][minI[3]-1][minI[4]];
				double val1 = y5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				
				minI[3]--;
				minDec[3] = ((tlat - val0) / (val1 - val0));
			}
			//ALT
			if (z[m] == alt[j])
			{
				minDec[4] = 0;
			} else if (alt[j] > z[m])
			{
				double val0 = z5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				double val1 = z5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]+1];
				minDec[4] = ((alt[j] - val0) / (val1 - val0));
			} else
			{
				double val0 = z5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]-1];
				double val1 = z5[minI[0]][minI[1]][minI[2]][minI[3]][minI[4]];
				
				minI[4]--;
				minDec[4] = ((alt[j] - val0) / (val1 - val0));
			}
			
			//initialize output
			output[i][j] = 0.0;

			//do the interpolation
			for(int ii=0;ii<=1;ii++) 
			{
				for(int jj=0;jj<=1;jj++) 
				{
					for(int kk=0;kk<=1;kk++) 
					{
						for(int ll=0;ll<=1;ll++) 
						{
							for (int mm=0; mm<=1; mm++)
							{
								//set the denspert indices
								int di[5] = {minI[0]+ii,minI[1]+jj,minI[2]+kk,minI[3]+ll,minI[4]+mm};
								
								if (di[0] >= indices[0])
								{
									di[0] = di[0] - indices[0];
									di[1]++;
								}
								if (di[1] >= indices[1])
								{
									di[1] = di[1] - indices[1];
									di[2]++;
								}
								if (di[2] >= indices[2])
								{
									di[2] = di[2] - indices[2];
									di[3]++;
								}
								if (di[3] >= indices[3])
								{
									di[3] = di[3] - indices[3];
									di[4]++;
								}
								if (di[4] >= indices[4])
								{
									printf("5D Interpolation Overflow.\n");
								}
								
								
								output[i][j] += (ii?minDec[0]:(1-minDec[0])) * (jj?minDec[1]:(1-minDec[1])) * \
										(kk?minDec[2]:(1-minDec[2])) * (ll?minDec[3]:(1-minDec[3])) * \
										(mm?minDec[4]:(1-minDec[4])) * \
										densPert[di[0]][di[1]][di[2]][di[3]][di[4]];
							}
						}
					}
				}
			}
		}

	}
	
	//close the DB
	sqlite3_close(dbFIRI);
	
	//free memory
	free(w);
	free(v);
	free(x);
	free(y);
	free(z);
	free(f10);
	free(f10data);
	free(dPert);
	for (int a=0; a<indices[0]; a++)
	{
		for (int b=0; b<indices[1]; b++)
		{
			for (int c=0; c<indices[2]; c++)
			{
				for (int d=0; d<indices[3]; d++) 
				{
					free(densPert[a][b][c][d]);
					free(w5[a][b][c][d]);
					free(v5[a][b][c][d]);
					free(x5[a][b][c][d]);
					free(y5[a][b][c][d]);
					free(z5[a][b][c][d]);
				}
				free(densPert[a][b][c]);
			}
			free(densPert[a][b]);
		}
		free(densPert[a]);
	}
	free(densPert);
	
	return output;
	
}