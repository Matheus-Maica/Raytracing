#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sqlite3.h"
#include "global.h"
#include "maxIndices.h"

static double output; //output time

//declare static function
static int callback(void *data, int argc, char **argv, char **azColName);

//sqlite3 callback function for data
static int callback(void *data, int argc, char **argv, char **azColName)
{   
	(void)data;
	(void)azColName;
	
	if (argv[0] == NULL)
	{
		output = 0.0;
	}else
	{
		output = atof(argv[0]);
	}

	
   return 0;
}

//routines to get the max time for given indices

double maxAE(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select MAX(TIME) from Hourly Where AEI IS NOT NULL AND AE != 99999";
		
	}else
	{
		sql = "Select MIN(TIME) from Hourly Where AEI IS NOT NULL AND AE != 99999";
	}	
		
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxDST(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select MAX(TIME) from Hourly Where DSTI IS NOT NULL AND DST != 99999";
		
	}else
	{
		sql = "Select MIN(TIME) from Hourly Where DSTI IS NOT NULL AND DST != 99999";
	}	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxKP(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select MAX(TIME) from THREEHOUR Where KP IS NOT NULL AND KP != 99999";
		
	}else
	{
		sql = "Select MIN(TIME) from THREEHOUR Where KP IS NOT NULL AND KP != 99999";
	}	
	
	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxAP(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select MAX(TIME) from THREEHOUR Where API IS NOT NULL AND AP != 99999";
		
	}else
	{
		sql = "Select MIN(TIME) from THREEHOUR Where API IS NOT NULL AND AP != 99999";
	}	
	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxPC(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select MAX(TIME) from HOURLY Where PC IS NOT NULL AND PC != 99999";
		
	}else
	{
		sql = "Select MIN(TIME) from HOURLY Where PC IS NOT NULL AND PC != 99999";
	}	
	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxF10(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select Field from METADATA Where NAME == 'F10_Max'";
		
	}else
	{
		sql = "Select MIN(TIME) from DAILY Where F10 IS NOT NULL AND F10 != 99999";
	}	
	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxF10F(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select MAX(TIME) from DAILY Where F10 IS NOT NULL AND F10 != 99999";
		
	}else
	{
		sql = "Select Field from METADATA Where NAME == 'F10_Max'";
	}	
	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxIG(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select Field from METADATA Where NAME == 'IG_Max'";
		
	}else
	{
		sql = "Select MIN(TIME) from MONTHLY Where IG IS NOT NULL AND IG != 99999";
	}	
	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}

double maxIG12(int mm,sqlite3 *db)
{
	//variables
	char *sql; //sql commands
	char cwd[1024], *zErrMsg; //directory, sql error
	int rc, nc=-1; //sqlite3 return code, null check
	
	if (db == NULL)
	{
		//setting directory
		strcpy(cwd,DIR);

		//get working directory and append database filename to the end
		strcat(cwd,"CHAIM_DB.db");

		//open the DB
		rc = sqlite3_open(cwd, &db);
		
		nc = 1;
	}

	if (mm >= 0)
	{
		sql = "Select MAX(TIME) from MONTHLY Where IG IS NOT NULL AND IG != 99999";
		
	}else
	{
		sql = "Select Field from METADATA Where NAME == 'IG_Max'";
	}	
	
	
	//Executing the command, getting the data
	rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
	if (rc != 0) printf("Error %i %s\n%s\n", rc, zErrMsg, sql);
	
	//close the DB
	if (nc == 1) sqlite3_close(db);

	return output;
}