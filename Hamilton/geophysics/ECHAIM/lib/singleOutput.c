#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "calcMF2.h"
#include "calcPert.h"
#include "calcHmF1.h"
#include "getDir.h"
#include "global.h"
#include "date.h"
#include "ECHAIM.h"
#include "getMagCoords.h"
#include "maxIndices.h"
#include "errorCodes.h"
#include "sqlite3.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//routine to get and return NmF2 for a given location and time
//arrays are the same length l0
double * NmF2(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err)
{
	//set variables
	double *output; //NmF2 output, length = l0
	double **r; //return from calcMF2
	double *jd, *mlat, *mlon; //array of julian dates, mag coords
	
	output = calloc(l0, sizeof(double));
	jd = calloc(l0, sizeof(double));
	mlat = (double *)calloc(l0, sizeof(double));
	mlon = (double *)calloc(l0, sizeof(double));
	
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
		jd[i] = julianDate(year[i], month[i], day[i], hour[i], min[i], sec[i]);
		
		//get mag coords
		double *m;
		
		m = getMagCoords(jd[i], lat[i], lon[i], 300.0, 1);
		mlat[i] = m[0];
		mlon[i] = m[1];
		
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
		
		for (int i=0; i<l0; i++)
		{
			ERRORCODES[i][0] = '-'; 
			ERRORCODES[i][1] = '-';
			if (mig == 0 || jd[i] > mig || jd[i] < minig) ERRORCODES[i][2] = 'C'; else ERRORCODES[i][2]=' ';
			if (mf10f == 0 || jd[i] > mf10f) ERRORCODES[i][3] = 'E'; else ERRORCODES[i][3]=' ';
			if (mf10 == 0 || jd[i] > mf10 || jd[i] < minf10) ERRORCODES[i][4] = 'F'; else ERRORCODES[i][4]=' ';
			if (mig12 == 0 || jd[i] > mig12) ERRORCODES[i][5] = 'G'; else ERRORCODES[i][5]=' ';
			if (mlat[i] < 50.0) ERRORCODES[i][6] = 'H'; else ERRORCODES[i][6]=' ';
			if (mlat[i] < 45.0) ERRORCODES[i][7] = 'I'; else ERRORCODES[i][7]=' ';
			ERRORCODES[i][8] = '-';
			ERRORCODES[i][9] = '-';
			
		}
	}
	
	//get the NmF2 output
	r = calcMF2(jd, lat, lon, mlat, mlon, l0, 1, db, dbCoefs);

	for (int i=0; i<l0; i++)
	{
		output[i] = r[0][i];
	}
	
	//close the DB
	sqlite3_close(db);
	sqlite3_close(dbCoefs);
	
	//free memory
	for (int i=0; i<2; i++) {free(r[i]);}
	free(r);
	free(jd);
	free(mlat);
	free(mlon);
	
	return output;
	
}

//routine to get and return NmF2 for a given location and time
//arrays are the same length l0
double * NmF2Storm(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err)
{
	//set variables
	double *output; //NmF2 output, length = l0
	double **r; //return from calcMF2
	double *jd, *mlat, *mlon; //array of julian dates, mag coords
	double *p; //perturbation results
	
	output = calloc(l0, sizeof(double));
	jd = calloc(l0, sizeof(double));
	mlat = (double *)calloc(l0, sizeof(double));
	mlon = (double *)calloc(l0, sizeof(double));
	
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
		jd[i] = julianDate(year[i], month[i], day[i], hour[i], min[i], sec[i]);
		
		//get mag coords
		double *m;
		
		m = getMagCoords(jd[i], lat[i], lon[i], 300.0, 1);
		mlat[i] = m[0];
		mlon[i] = m[1];
		
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
			if (mae == 0 || jd[i] > mae) ERRORCODES[i][0] = 'A'; \
				else if (jd[i] >= 2442778.5 && jd[i] <= 2443509.5) ERRORCODES[i][0] = 'A'; else ERRORCODES[i][0]=' ';
			if (mpc == 0 || jd[i] > mpc) ERRORCODES[i][1] = 'B'; \
				else if (jd[i] >= 2447161.5 && jd[i] <= 2442778.5) 
				{
					ERRORCODES[i][0] = 'A';
					ERRORCODES[i][1] = 'B';
					
				}else ERRORCODES[i][0]=' ';
			if (mig == 0 || jd[i] > mig || jd[i] < minig) ERRORCODES[i][2] = 'C'; else ERRORCODES[i][2]=' ';
			if (mf10f == 0 || jd[i] > mf10f) ERRORCODES[i][3] = 'E'; else ERRORCODES[i][3]=' ';
			if (mf10 == 0 || jd[i] > mf10 || jd[i] < minf10) ERRORCODES[i][4] = 'F'; else ERRORCODES[i][4]=' ';
			if (mig12 == 0 || jd[i] > mig12) ERRORCODES[i][5] = 'G'; else ERRORCODES[i][5]=' ';
			if (mlat[i] < 50.0) ERRORCODES[i][6] = 'H'; else ERRORCODES[i][6]=' ';
			if (mlat[i] < 45.0) ERRORCODES[i][7] = 'I'; else ERRORCODES[i][7]=' ';
			if (map == 0 || jd[i] > map || jd[i] < minap) ERRORCODES[i][8] = 'J'; else ERRORCODES[i][8]=' ';
			if (mdst == 0 || jd[i] > mdst || jd[i] < mindst) ERRORCODES[i][9] = 'K'; else ERRORCODES[i][9]=' ';
			
		}
	}
	
	
	//get the NmF2 output
	r = calcMF2(jd, lat, lon, mlat, mlon, l0, 1, db, dbCoefs);
	
	//get the perturbation values
	p = calcPert(jd, mlat, mlon, l0, db, dbCoefs);
	
		
	for (int i=0; i<l0; i++)
	{
		output[i] = r[0][i] * pow(10.0, p[i]);
	}
		
	//close the DB
	sqlite3_close(db);
	sqlite3_close(dbCoefs);
		
	//free memory
	free(jd);
	for (int i=0; i<2; i++) {free(r[i]);}
	free(r);
	free(p);
	free(mlat);
	free(mlon);

	return output;
	
}

//routine to get and return HmF2 for a given location and time
//arrays are the same length l0
double * HmF2(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err)
{
	//set variables
	double *output; //NmF2 output, length = l0
	double **r; //return from calcMF2
	double *jd, *mlat, *mlon; //array of julian dates, mag coords
	
	output = calloc(l0, sizeof(double));
	jd = calloc(l0, sizeof(double));
	mlat = (double *)calloc(l0, sizeof(double));
	mlon = (double *)calloc(l0, sizeof(double));
	
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
		jd[i] = julianDate(year[i], month[i], day[i], hour[i], min[i], sec[i]);
		
		//get mag coords
		double *m;
		
		m = getMagCoords(jd[i], lat[i], lon[i], 300.0, 1);
		mlat[i] = m[0];
		mlon[i] = m[1];
				
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
		
		for (int i=0; i<l0; i++)
		{
			ERRORCODES[i][0] = '-'; 
			ERRORCODES[i][1] = '-';
			if (mig == 0 || jd[i] > mig || jd[i] < minig) ERRORCODES[i][2] = 'C'; else ERRORCODES[i][2]=' ';
			if (mf10f == 0 || jd[i] > mf10f) ERRORCODES[i][3] = 'E'; else ERRORCODES[i][3]=' ';
			if (mf10 == 0 || jd[i] > mf10 || jd[i] < minf10) ERRORCODES[i][4] = 'F'; else ERRORCODES[i][4]=' ';
			if (mig12 == 0 || jd[i] > mig12) ERRORCODES[i][5] = 'G'; else ERRORCODES[i][5]=' ';
			if (mlat[i] < 50.0) ERRORCODES[i][6] = 'H'; else ERRORCODES[i][6]=' ';
			if (mlat[i] < 45.0) ERRORCODES[i][7] = 'I'; else ERRORCODES[i][7]=' ';
			ERRORCODES[i][8] = '-';
			ERRORCODES[i][9] = '-';
			
		}
	}
	
	
	//get the HmF2 output
	r = calcMF2(jd, lat, lon, mlat, mlon, l0, 2, db, dbCoefs);

	for (int i=0; i<l0; i++)
	{
		output[i] = r[1][i];
	}
	
	//close the DB
	sqlite3_close(db);
	sqlite3_close(dbCoefs);
	
	//free memory
	for (int i=0; i<2; i++) {free(r[i]);}
	free(r);
	free(jd);
	free(mlat);
	free(mlon);
	
	return output;
	
}

//routine to get and return HmF1 for a given location and time
//arrays are the same length l0
double * HmF1(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err)
{
	//set variables
	double *r; //return from calcMF2
	double *jd, *mlat, *mlt; //array of julian dates, mag coords
	
	jd = calloc(l0, sizeof(double));
	mlat = (double *)calloc(l0, sizeof(double));
	mlt = (double *)calloc(l0, sizeof(double));
	
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
		jd[i] = julianDate(year[i], month[i], day[i], hour[i], min[i], sec[i]);
		
		//get mag coords
		double *m;
		
		m = getMagCoords(jd[i], lat[i], lon[i], 300.0, 1);
		mlat[i] = m[0];
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
		
		for (int i=0; i<l0; i++)
		{
			ERRORCODES[i][0] = '-'; 
			ERRORCODES[i][1] = '-';
			if (mig == 0 || jd[i] > mig || jd[i] < minig) ERRORCODES[i][2] = 'C'; else ERRORCODES[i][2]=' ';
			if (mf10f == 0 || jd[i] > mf10f) ERRORCODES[i][3] = 'E'; else ERRORCODES[i][3]=' ';
			if (mf10 == 0 || jd[i] > mf10 || jd[i] < minf10) ERRORCODES[i][4] = 'F'; else ERRORCODES[i][4]=' ';
			if (mig12 == 0 || jd[i] > mig12) ERRORCODES[i][5] = 'G'; else ERRORCODES[i][5]=' ';
			if (mlat[i] < 50.0) ERRORCODES[i][6] = 'H'; else ERRORCODES[i][6]=' ';
			if (mlat[i] < 45.0) ERRORCODES[i][7] = 'I'; else ERRORCODES[i][7]=' ';
			ERRORCODES[i][8] = '-';
			ERRORCODES[i][9] = '-';
			
		}
	}
	
	
	//get the HmF2 output
	r = calcHmF1(jd, lat, lon, mlat, mlt, l0, db, dbCoefs);
	
	//close the DB
	sqlite3_close(db);
	sqlite3_close(dbCoefs);
	
	//free memory
	free(jd);
	free(mlat);
	free(mlt);
	
	return r;
	
}