#include <stdio.h>
#include <stdlib.h>
#include <curl/curl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "ECHAIM.h"
#include "getDir.h"
#include "global.h"

//This function checks the age of the local DB
//if the DB file is older than the last expected update time, the DB file
//available on the ECHAIM website is downloaded and will overwrite the current DB file
//
//Output is used to know what the function did
//0 = database file was updated after updateTime
//1 = current time is less than updateTime
//2 = current time is later than updateTime AND DB file was not updated after updateTime
//   The DB file was updated
//3 = DB file was forced to update without any checks

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

int updateLocalDB (int force)
{
	struct stat buffer; //status output structure
	int status, rc=0; //stat status
	struct tm fileTime, currentTime, updateTime; //time structures
	time_t now, fileT, currentT, updateT; //time_t for current time
	char dirDB[1024];
	
	//set the directory from the config file
	getDir();
	
	//set file name and directory
	strcpy(dirDB,DIR);
	strcat(dirDB, "CHAIM_DB.db"); //DB file path
	
	//User asked to force an update (download the DB file without any checks)
	if (force)
	{
		//download
		FILE* file = fopen(dirDB, "wb+"); //the file pointer

		if (!file) {
	            perror("Error opening file");
                    return -1;
        	}
		
		char url[] = "https://chain-new.chain-project.net/echaim_downloads/DBFiles/CHAIM_DB.db";
		
		//initialize curl command, set options, and close
		CURL* easyhandle = curl_easy_init();
		rc = curl_easy_setopt(easyhandle, CURLOPT_URL, url);
		rc = curl_easy_setopt(easyhandle, CURLOPT_WRITEFUNCTION, write_data);
		rc = curl_easy_setopt(easyhandle, CURLOPT_WRITEDATA, file);
		//curl_easy_setopt(easyhandle, CURLOPT_VERBOSE, 1L); // Verbose output for debugging

		rc = curl_easy_perform(easyhandle);
		if (rc != CURLE_OK) {
            	    fprintf(stderr, "CURL error: %s\n", curl_easy_strerror(rc));
        	}

		curl_easy_cleanup(easyhandle);	
		fclose(file); //close the file

		printf("\nThe DB file was forced to update by the user.\n");
		return 3;
	}
	
	//get file status
	status = stat(dirDB, &buffer);
	
	//If the DB file is missing, download it
	if (status < 0)
	{
		//download
		FILE* file = fopen(dirDB, "wb+"); //the file pointer
		char url[] = "https://chain-new.chain-project.net/echaim_downloads/DBFiles/CHAIM_DB.db";
		
		//initialize curl command, set options, and close
		CURL* easyhandle = curl_easy_init();
		rc = curl_easy_setopt(easyhandle, CURLOPT_URL, url);
		rc = curl_easy_setopt(easyhandle, CURLOPT_WRITEFUNCTION, write_data);
		rc = curl_easy_setopt(easyhandle, CURLOPT_WRITEDATA, file);
		rc = curl_easy_perform(easyhandle);
		curl_easy_cleanup(easyhandle);
	
		fclose(file); //close the file

		printf("\nThe DB file has been updated.\n");
		return 2;
	}
	
	//get file modified time
	fileTime = *gmtime(&buffer.st_mtime); //(&buffer.st_mtim.tv_sec);

	//get current time	
	time(&now);
	currentTime = *gmtime(&now);
	
	//set updateTime
	updateTime = currentTime;
	updateTime.tm_hour = 11; //set to 11:30 GMT (8:30 ADT)
	updateTime.tm_min = 30;
	
	//convert to time_t
	currentT = mktime(&currentTime);
	updateT = mktime(&updateTime);
	fileT = mktime(&fileTime);
	
	//The DB file was last update after updateTime
	if (difftime(fileT, updateT) > 0) //if returns > 0 then first input is greater than second
	{
		printf("\nThe DB file has already been updated today.\n");
		return 0;
	}
	
	//Current time is before updateTime
	if (difftime(updateT, currentT) > 0)
	{
		printf("\nThe website has not updated the DB file yet today.\n");
		return 1;
	}
	
	//download
	FILE* file = fopen(dirDB, "wb+"); //the file pointer
	char url[] = "https://chain-new.chain-project.net/echaim_downloads/DBFiles/CHAIM_DB.db";
	
	//initialize curl command, set options, and close
	CURL* easyhandle = curl_easy_init();
	rc = curl_easy_setopt(easyhandle, CURLOPT_URL, url);
	rc = curl_easy_setopt(easyhandle, CURLOPT_WRITEFUNCTION, write_data);
	rc = curl_easy_setopt(easyhandle, CURLOPT_WRITEDATA, file);
	rc = curl_easy_perform(easyhandle);
	curl_easy_cleanup(easyhandle);

	fclose(file); //close the file
	
	printf("\nThe DB file has been updated.\n");
	return 2;
}
