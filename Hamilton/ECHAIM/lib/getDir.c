#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "global.h"
#include "getDir.h"
#include "aacgm_init.h"
#include "igrf_init.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Global variables
char DIR[1024]; //Directory

//gets the directory from the config file and puts it in the dir variable
//This will get the directory from the config file, make the directory if necessary
//and set it to the global DIR
//If the config file doesn't exist, then ask the user for a directory, write it to a config
//in the CWD
int getDir()
{
	// Already initialized (via echaim_init(), or an earlier call in this
	// process) -- skip re-reading configECHAIM.dat entirely. Without this
	// guard, every public ECHAIM call re-enters this function, and a
	// missing configECHAIM.dat in the caller's CWD would fall through to
	// the interactive fgets(stdin) below -- fatal for a library embedded
	// in another process (e.g. called from Python via ctypes).
	if (DIR[0] != '\0') return 1;

	char str[128];
	size_t len = 128;
	FILE *file = NULL;

	file = fopen("configECHAIM.dat", "r");
	
	if (file == NULL)
	{
				printf("Configuration file not found\n");
		printf("Enter the directory which contains the ECHAIM requirement files: ");
		
		fgets(str, len, stdin);
	
		file = fopen("configECHAIM.dat", "w+");
	
		fprintf(file, "#This is the directory which contains the ECHAIM database (.DB) file,\n");
		fprintf(file, "#the AACGM coefficient file, the COEFS database, and downloaded indices.\n");
		fprintf(file, "%s", str);
		
		fclose(file);

	} else
	{
	
		//read through the configuration file
		while ((fgets(str, len, file)) != NULL)
		{
			//Skip header and comments
			if (str[0] == '#') continue;
			
			fclose(file);
			
			break;
		}
	}
	
	//putting the directory in dataDir
	int n = strcspn(str, "\n");
	str[n] = 0;	
	
	if (str[n-1] == '\r') 
	{
		str[n-1] = 0;
		n--;
	}
	
	if (str[n] != '/') {str[n] = '/';}

	strcpy(DIR,str);

	//initialize the env variables for AACGM
	int init_err = aacgm_init(DIR);

	if (init_err) {
        fprintf(stderr, "Failed to initialize AACGM coefficients.\n");
    }

	//initialize the IGRF coefficients
	igrf_init();

	return 1;
}

// See getDir.h: sets DIR directly (bypassing configECHAIM.dat/stdin) and
// runs the same AACGM/IGRF init getDir() would otherwise trigger. Call once
// before any other ECHAIM function.
int echaim_init(const char *dir)
{
	size_t n = strlen(dir);
	if (n == 0 || n >= sizeof(DIR) - 1) return -1;

	strcpy(DIR, dir);
	if (DIR[n - 1] != '/')
	{
		DIR[n] = '/';
		DIR[n + 1] = '\0';
	}

	int init_err = aacgm_init(DIR);
	if (init_err)
	{
		fprintf(stderr, "Failed to initialize AACGM coefficients.\n");
		return init_err;
	}

	igrf_init();

	return 0;
}