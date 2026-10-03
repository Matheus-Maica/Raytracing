#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

double *getH0(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array)
	tempCallback = calloc(264, sizeof(double)); //there are 264 H0 coefs
	
	//SQLite3 command to get the coefs
	sql = "Select * from H0";
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}

double *getHE(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code

	//set the global pointer (output array)
	tempCallback = calloc(264, sizeof(double)); //there are 264 HE coefs
	
	//SQLite3 command to get the coefs
	sql = "Select * from HE";
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}

double *getHF1(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array)
	tempCallback = calloc(264, sizeof(double)); //there are 264 HF1 coefs
	
	//SQLite3 command to get the coefs
	sql = "Select * from HF1";
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}

double *getHMF1(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array)
	tempCallback = calloc(126, sizeof(double)); //there are 126 HMF1 coefs
	
	//SQLite3 command to get the coefs
	sql = "Select * from HMF1";
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}

double **getHMF2(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array) ***
	tempCallbackMF2 = (double **) calloc(24, sizeof(double)); //there are 184x24 HMF1 coefs
	
	for (int i=0; i<24; i++)
	{
		tempCallbackMF2[i] = (double *) calloc(184, sizeof(double));
	}
	
	//SQLite3 command to get the coefs
	sql = "Select * from HMF2";
	
	//set the callback counters to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callbackMF2, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallbackMF2;
}

double **getNMF2(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array) ***
	tempCallbackMF2 = (double **) calloc(24, sizeof(double)); //there are 567x24 HMF1 coefs
	
	for (int i=0; i<24; i++)
	{
		tempCallbackMF2[i] = (double *) calloc(568, sizeof(double));
	}
	
	//SQLite3 command to get the coefs
	sql = "Select * from NMF2";
	
	//set the callback counters to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callbackMF2, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallbackMF2;
}

double *getNe(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array)
	tempCallback = calloc(742, sizeof(double)); //there are 264 H0 coefs
	
	//SQLite3 command to get the coefs
	sql = "Select * from Ne";
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}

double **getPERT(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array) ***
	tempCallbackMF2 = (double **) calloc(24, sizeof(double)); //there are 361x24 PERT coefs
	
	for (int i=0; i<24; i++)
	{
		tempCallbackMF2[i] = (double *) calloc(361, sizeof(double));
	}
	
	//SQLite3 command to get the coefs
	sql = "Select * from PERT";
	
	//set the callback counters to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callbackMF2, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallbackMF2;
}

double *getAE(sqlite3 *db)
{
	//declare variables
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//set the global pointer (output array)
	tempCallback = calloc(7, sizeof(double));
	
	//SQLite3 command to get the coefs
	sql = "Select * from AE";
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	return tempCallback;
}
