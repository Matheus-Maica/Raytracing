#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "sqlite3.h"
#include "date.h"
#include "ECHAIM.h"
#include "getDir.h"
#include "global.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

static char tempCallback[64]; //pointers to the indices to be returned by the functions

//declare static functions
static int callback(void *data, int argc, char **argv, char **azColName);


//sqlite3 callback function for data
//This is for all the vector coefs
//H0, HE, HF1, HMF1
static int callback(void *data, int argc, char **argv, char **azColName)
{   
	(void)data;
	(void)azColName;
	
	strcpy(tempCallback,argv[0]);

	return 0;
}

double getDBDate()
{
	//declare variables
	double output;
	char *sql; //sql commands
	char *zErrMsg; //directory, sql error
	sqlite3 *db; //SQLite3 database variable
	
	//set the directory from the config file
	getDir();
	
	//sqlite3 variables
	char cwd[1024]; //directory, sql error

	//setting directory
	strcpy(cwd,DIR);
	
	//get working directory and append database filename to the end
	strcat(cwd,"CHAIM_DB.db");
	
	//open the DB
	int rc = sqlite3_open(cwd, &db);
	
	//SQLite3 command to get the coefs
	sql = "Select Field from Metadata Where Name == \'Update\'";
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	sqlite3_close(db);
	
	output = atof(tempCallback);
	
	return output;
}