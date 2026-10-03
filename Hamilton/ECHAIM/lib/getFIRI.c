#include <stdlib.h>
#include <stdio.h>
#include "getFIRI.h"

//Functions to get the D-region DB data

static double *tempCallback; //pointer to the indices to be returned by the functions
static int cbc; //callback counter

//declare static function
static int callbackGetIndices(void *data, int argc, char **argv, char **azColName);

//sqlite3 callback function for data
static int callbackGetIndices(void *data, int argc, char **argv, char **azColName)
{   
	(void)data;
	(void)azColName;
	
	tempCallback[cbc] = (argv[0] == NULL) ? 0.0/0.0 : atof(argv[0]);
	cbc++;

   return 0;
}

//returns the (1D) array of the FIRI DB data
//type is the array to return:
//w,v,x,y,z,denspertextra
double *getFIRI(char *type, sqlite3 *db)
{
	//variables
	char *sql, b[128]; //sql commands amd buffer
	char *zErrMsg; //directory, sql error
	int rc; //sqlite3 return code
	
	//the length of the arrays
	long l0 = 139986;
	
	//set the output pointer to the global variable
	//this is necessary for the callback function
	tempCallback = calloc(l0, sizeof(double));
	
	//create the SQL command
	sql = "Select * from %s";
	sprintf(b,sql,type);
	
	//set the callback counter to 0
	cbc = 0;
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, b, callbackGetIndices, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, b);
	
	return tempCallback;
}
