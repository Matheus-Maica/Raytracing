#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
A: Measured AE index was not available. A synthetic AE index based on PC index has been used instead for these data points if available.
B: PC index is not available. Synthetic AE index is generated using PC = 0.0 for these data points.
C: Data requested is beyond or prior to the IG index record. IG12 has been used if available.
E: These requested data periods are for a time past the maximum time stamp of the NOAA F10.7 flux forecast. 
F: Measured F10.7 index was not available. The corresponding output was generated using a NOAA forecast of F10.7 flux if available.
G: Data requested is beyond IG12 index record.
H: Warning - Requested location is below the lower boundary of the model (50N geomagnetic). Since MLat is above 45N geomagnetic output is still provided.
   Output should be interpreted with caution.
I: Warning - Requested location is below 45N geomagnetic latitude. Output has been forced to NAN.  
J: Requested period is beyond or prior to the times available in the ap reference database. ap has been set to the all-time median (9.1)
K: Requested period is beyond or prior to the times available in the dst reference database. dst has been set to the all-time median (-10.2) 

*/


//declare errorcodes array
char **ERRORCODES = NULL;
int l0last;

//The error codes will be a 2d array
//[number or outputs (locations,times, etc), 10 (number or error characters plus string terminator)]

//Initialize the array
//l0 is number or inputted times/locations etc
//**This array must be freed at the end of the program**
void logErrors(int l0)
{

        // If previously set up then free the pointers
	
	if (ERRORCODES != NULL) 
	{   
	    for (int i=0; i<l0last; i++)
	    {
		free(ERRORCODES[i]);
	    }
	    
	    free(ERRORCODES);
	    ERRORCODES = NULL;
	}

        // If l0 is <= 0 then return without setting new pointers. Useful for tidying up at the end.
		
	if (l0 <= 0)
	{    
	     return;
	}

	//initiate the arrays
	
	// Note that the first level pointers need to be sized for pointers
	ERRORCODES = (char **) calloc(l0, sizeof(char *));
	
	//this finishes allocating the 2D output array
	for (int i=0; i<l0; i++)
	{
		ERRORCODES[i] = (char *) calloc(11, sizeof(char));
		ERRORCODES[i][10] = '\0';
	}
	l0last = l0;
}

//Allows the user to get the global error codes array
char **getErrors ()
{
	if (ERRORCODES == NULL)
	{
		return NULL;
	}else
	{
		return ERRORCODES;
	}	
}
