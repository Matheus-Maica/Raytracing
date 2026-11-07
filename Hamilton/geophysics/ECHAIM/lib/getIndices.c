#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sqlite3.h"
#include "date.h"
#include "global.h"
#include "getIndices.h"
#include "mathlib.h"
#include "maxIndices.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Routines to retrieve the indices from the DB for a given time range

static double *tempCallback; //pointer to the indices to be returned by the functions
static double *timeReturn;
static int cbc; //callback counter

//declare static function
static int callbackGetIndices(void *data, int argc, char **argv, char **azColName);

//sqlite3 callback function for data
static int callbackGetIndices(void *data, int argc, char **argv, char **azColName)
{   
	(void)data;
	(void)azColName;
	
	if (argv[1] == NULL)
	{
		tempCallback[cbc] = 0.0/0.0;
		timeReturn[cbc] = atof(argv[0]);
		cbc++;
	} else
	{
		tempCallback[cbc] = (atof(argv[1]) == 99999) ? 0.0/0.0 : atof(argv[1]);
		timeReturn[cbc] = atof(argv[0]);
		cbc++;
	}
  
   return 0;
}

//routine to get the AE data
//Expects pointer to array to hold the data and the length
//start and end julian dates
double *AE (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code

	//set the output pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 24), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 24), sizeof(double));

	sql = "Select TIME,AEI from Hourly Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

//routine to get the AE data
//Expects pointer to array to hold the data and the length
//start and end julian dates
double *AEraw (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	

	//set the output pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 24), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 24), sizeof(double));

	sql = "Select TIME,AE from Hourly Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *DST (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 24), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 24), sizeof(double));

	sql = "Select TIME,DSTI from Hourly Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *KP (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));

	sql = "Select TIME,KP from THREEHOUR Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *AP (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));

	sql = "Select TIME,API from THREEHOUR Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *AP_8 (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));

	sql = "Select TIME,AP_8 from THREEHOUR Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *APraw (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 8), sizeof(double));

	sql = "Select TIME,AP from THREEHOUR Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *F10 (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the output pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1)), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1)), sizeof(double));

	sql = "Select TIME,F10 from DAILY Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
		
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *IG (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)(((jdMonthly(jd1) - jdMonthly(jd0)) / 30.0) + 2), sizeof(double));
	timeReturn = calloc((int)(((jdMonthly(jd1) - jdMonthly(jd0)) / 30.0) + 2), sizeof(double));

	sql = "Select * from MONTHLY Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jdMonthly(jd0), jdMonthly(jd1)+1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *PC (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 24.0), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 24.0), sizeof(double));

	sql = "Select TIME,PCI from HOURLY Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *PCraw (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the ouput pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1) * 24.0), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1) * 24.0), sizeof(double));

	sql = "Select TIME,PC from HOURLY Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *F10_27 (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the output pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1)), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1)), sizeof(double));

	sql = "Select TIME,F10_27 from DAILY Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
		
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}

double *F10_81 (double jd0, double jd1, double **x, int **l0, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the output pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc((int)((jd1 - jd0 + 1)), sizeof(double));
	timeReturn = calloc((int)((jd1 - jd0 + 1)), sizeof(double));

	sql = "Select TIME,F10_81 from DAILY Where Time Between ";
	sprintf(b, "%s%.6f And %.6f",sql, jd0, jd1);

	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
		
	*x = timeReturn;
	*l0 = &cbc;
	
	return tempCallback;
	
}
