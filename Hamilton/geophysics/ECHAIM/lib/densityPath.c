#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calcMF2.h"
#include "getDir.h"
#include "global.h"
#include "date.h"
#include "densityFunc.h"
#include "calcHmF1.h"
#include "calcBot.h"
#include "calcTop.h"
#include "calcPert.h"
#include "calcPrecip.h"
#include "ECHAIM.h"
#include "getMagCoords.h"
#include "maxIndices.h"
#include "errorCodes.h"
#include "sqlite3.h"
#include <string.h>
#include "ECHAIMFIRI2018.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************


//returns density along a given arc
//inputs are latitudes, longitudes, altitudes, times (length l0)
//output is an array [legnth l0]
double * densityPath(double *lat, double *lon, double *alt, \
						double *year, double *month, double *day, \
						double *hour, double *min, double *sec, int storm, int precip, int dregion, int l0, int err)
{
	//set variables
	double *output; //density profile output, length = [l0,l1,l2]
	double **rMF2, *rHmF1, **rBot, *rTop; //return from calcs
	double *jd, *mlat, *mlon, *mlt; //inputs to calcs
	double *rPert; //pert output
	double **rPre, **rDreg; //precip model output, d region
	
	//allocate memory for output
	output = (double *) calloc(l0, sizeof(double));
	jd = (double *) calloc(l0, sizeof(double));
	mlat = (double *) calloc(l0, sizeof(double));
	mlon = (double *) calloc(l0, sizeof(double));
	mlt = (double *) calloc(l0, sizeof(double));

	//set the directory from the config file
	getDir();
	
	//sqlite3 variables
	char cwd[1024]; //directory, sql error
	sqlite3 *db, *dbCoefs; //SQLite3 database variable

	//setting directory
	strcpy(cwd,DIR);
	
	//get working directory and append database filename to the end
	strcat(cwd,"CHAIM_DB.db");
	
	//open the DB
	int rc = sqlite3_open(cwd, &db);
	if (rc) {printf("Error: CHAIM_DB could not be opened\n");}
	
	//setting directory
	strcpy(cwd,DIR);
	
	//get working directory and append database filename to the end
	strcat(cwd,"COEFS_DB.db");
	
	//open the DB
	rc = sqlite3_open(cwd, &dbCoefs);
	if (rc) {printf("Error: COEFS_DB could not be opened\n");}
	
	//set error logging
	if (err) logErrors(l0);

	for (int i=0; i<l0; i++)
	{
		//set the inputs
		jd[i] = julianDate(year[i], month[i], day[i], hour[i], min[i], sec[i]);
		
		//get mag coords
		double *m;
		
		m = getMagCoords(jd[i], lat[i], lon[i], 300.0, 1);
		mlat[i] = m[0];
		mlon[i] = m[1];
		mlt[i] = m[2];
				
		free(m);
	}
	
	//Setting the error codes
	if (ERRORCODES != NULL)
	{
		double mf10 = maxF10(1,NULL);
		double mf10f = maxF10F(1,NULL);
		double minf10 = maxF10(-1,NULL);
		double mig = maxIG(1,NULL);
		double mig12 = maxIG12(1,NULL);
		double minig = maxIG(-1,NULL);
		double mae = maxAE(1,NULL);
		double mpc = maxPC(1,NULL);
		double mdst = maxDST(1,NULL);
		double map = maxAP(1,NULL);
		double mindst = maxDST(-1,NULL);
		double minap = maxAP(-1,NULL);
		
		for (int i=0; i<l0; i++)
		{
			
			if (mig == 0 || jd[i] > mig || jd[i] < minig) ERRORCODES[i][2] = 'C'; else ERRORCODES[i][2]=' ';
			if (mf10f == 0 || jd[i] > mf10f) ERRORCODES[i][3] = 'E'; else ERRORCODES[i][3]=' ';
			if (mf10 == 0 || jd[i] > mf10 || jd[i] < minf10) ERRORCODES[i][4] = 'F'; else ERRORCODES[i][4]=' ';
			if (mig12 == 0 || jd[i] > mig12) ERRORCODES[i][5] = 'G'; else ERRORCODES[i][5]=' ';
			if (mlat[i] < 50.0) ERRORCODES[i][6] = 'H'; else ERRORCODES[i][6]=' ';
			if (mlat[i] < 45.0) ERRORCODES[i][7] = 'I'; else ERRORCODES[i][7]=' ';
			
			if (storm)
			{
				if (mae == 0 || jd[i] > mae) ERRORCODES[i][0] = 'A'; \
					else if (jd[i] >= 2442778.5 && jd[i] <= 2443509.5) ERRORCODES[i][0] = 'A'; else ERRORCODES[i][0]=' ';
				if (mpc == 0 || jd[i] > mpc) ERRORCODES[i][1] = 'B'; \
				else if (jd[i] >= 2447161.5 && jd[i] <= 2442778.5) 
				{
					ERRORCODES[i][0] = 'A';
					ERRORCODES[i][1] = 'B';
					
				}else ERRORCODES[i][0]=' ';
				if (map == 0 || jd[i] > map || jd[i] < minap) ERRORCODES[i][8] = 'J'; else ERRORCODES[i][8]=' ';
				if (mdst == 0 || jd[i] > mdst || jd[i] < mindst) ERRORCODES[i][9] = 'K'; else ERRORCODES[i][9]=' ';
			}else
			{
				ERRORCODES[i][0] = '-';
				ERRORCODES[i][1] = '-';
				ERRORCODES[i][8] = '-';
				ERRORCODES[i][9] = '-';
			}
			
		}
	}
				
	//get the NmF2 and HmF2 output [0][] = N, [1][0] = H
	rMF2 = calcMF2(jd, lat, lon, mlat, mlon, l0, 0, db, dbCoefs);
	
	//get HmF1
	rHmF1 = calcHmF1(jd, lat, lon, mlat, mlt, l0, db, dbCoefs);
	
	//get bottomside components [0][] = HF1, [1][] = HE, [2][] = H0
	rBot = calcBot(jd, lat, lon, mlat, mlt, l0, db, dbCoefs);
	
	//get topside component
	rTop = calcTop(jd, lat, lon, mlat, mlt, l0, db, dbCoefs);
	
	if (storm)
	{
		rPert = calcPert(jd, mlat, mlon, l0, db, dbCoefs);
	}
	
	if (precip)
	{
		rPre = calcPrecip(jd, lat, lon, l0, alt, l0, db);
	}
	
	if (dregion)
	{
		rDreg = ECHAIMFIRI2018(jd, lat, lon, l0, alt, l0, db);
	}
	
	
	for (int i=0; i<l0; i++)
	{
		if (storm)
		{
			rMF2[0][i] *= pow(10.0, rPert[i]);
		}
		
		
		//if altitude is le HmF2
		if (alt[i] <= rMF2[1][i])
		{
			output[i] = rMF2[0][i] * botFunc(alt[i], rMF2[1][i], rHmF1[i], 102.0,rBot[2][i], rBot[0][i], rBot[1][i]);
		}
		else
		{
			output[i] = rMF2[0][i] * topFunc(alt[i], rMF2[1][i], rTop[i]);
		}
		
		if (dregion)
		{
			output[i] =  output[i] + rDreg[i][0];
		}
		
		if (precip)
		{	
			output[i] =  sqrt(pow(output[i] / 1e6, 2.0) + rPre[i][0]) * 1e6;
		}
	}
	
	//close the DB
	sqlite3_close(db);
	sqlite3_close(dbCoefs);
	
	for (int i=0; i<2; i++) {free(rMF2[i]);}
	free(rMF2);
	free(rHmF1);
	for (int i=0; i<3; i++) {free(rBot[i]);}
	free(rBot);
	free(rTop);
	free(jd);
	free(mlat);
	free(mlon);
	free(mlt);
	if (storm) {free(rPert);}
	if (precip) {free(rPre);}
		

	
	return output;
	
}