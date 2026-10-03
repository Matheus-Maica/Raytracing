#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sqlite3.h"
#include "global.h"
#include "getCoefs.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

static double *tempCallback, **tempCallbackMF2; //pointers to the indices to be returned by the functions
static int cbc; //callback counter

//declare static functions
static int callback(void *data, int argc, char **argv, char **azColName);
static int callbackMF2(void *data, int argc, char **argv, char **azColName);


//sqlite3 callback function for data
//This is for all the vector coefs
//H0, HE, HF1, HMF1
static int callback(void *data, int argc, char **argv, char **azColName)
{   
	(void)data;
	(void)azColName;
	
	for(int i = 0; i<argc; i++)
	{
	  tempCallback[cbc] = atof(argv[i]);
	  cbc++;
	}

	return 0;
}

//This is for all the matrix coefs
//HMF2, NMF2
static int callbackMF2(void *data, int argc, char **argv, char **azColName)
{
	(void)data;
	(void)azColName;
	
	for(int i = 0; i<argc; i++)
	{
		tempCallbackMF2[i][cbc] = atof(argv[i]);
	}

	cbc++;

	return 0;
}

double *getPrecipAEMean(sqlite3 *db, int isae)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array)
	int l0 = 23;
	if (!isae) {l0 = 26;}
	tempCallback = calloc(l0, sizeof(double));
	
	//SQLite3 command to get the coefs
	if (isae)
	{
		sql = "Select mean from aebins";
	} else
	{
		sql = "Select mean from pcbins";
	}
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}

double **getPrecipMLT(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//hardcoded dimensions
	int l0 = 50;
	int l1 = 100;
	
	//set the global pointer (output array) ***
	tempCallbackMF2 = (double **) calloc(l0, sizeof(double)); 
	
	for (int i=0; i<l0; i++)
	{
		tempCallbackMF2[i] = (double *) calloc(l1, sizeof(double));
	}
	
	//SQLite3 command to get the coefs
	sql = "Select * from outmlts";
	
	//set the callback counters to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callbackMF2, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallbackMF2;
}

double **getPrecipMLAT(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//hardcoded dimensions
	int l0 = 50;
	int l1 = 100;
	
	//set the global pointer (output array) ***
	tempCallbackMF2 = (double **) calloc(l0, sizeof(double)); //there are 567x24 HMF1 coefs
	
	for (int i=0; i<l0; i++)
	{
		tempCallbackMF2[i] = (double *) calloc(l1, sizeof(double));
	}
	
	//SQLite3 command to get the coefs
	sql = "Select * from outmlats";
	
	//set the callback counters to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callbackMF2, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallbackMF2;
}

double *getPrecipEnergy(int row, int col, int tab, sqlite3 *db, int isae)
{
	//declare variables
	char sql[256], *b; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//sqlite row ID starts with 1, so increase input row
	row++;
	
	//set the global pointer (output array)
	tempCallback = calloc(1, sizeof(double));
	
	//SQLite3 command to get the coefs
	if (isae)
	{
		b = "SELECT \"%i\" from output_gridded_energy_ae%i WHERE ROWID = %i;";
	} else
	{
		b = "SELECT \"%i\" from output_gridded_energy_pc%i WHERE ROWID = %i;";
	}
	sprintf(sql, b, col, tab, row);
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}

double *getPrecipFlux(int row, int col, int tab, sqlite3 *db, int isae)
{
	//declare variables
	char sql[256], *b; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//sqlite row ID starts with 1, so increase input row
	row++;
	
	//set the global pointer (output array)
	tempCallback = calloc(1, sizeof(double));
	
	//SQLite3 command to get the coefs
	if (isae)
	{
		b = "SELECT \"%i\" from output_gridded_flux_ae%i WHERE ROWID = %i;";
	} else
	{
		b = "SELECT \"%i\" from output_gridded_flux_pc%i WHERE ROWID = %i;";
	}
	sprintf(sql, b, col, tab, row);
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}
