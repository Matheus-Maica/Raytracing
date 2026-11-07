#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include "nrlmsise00.h"
#include "runNRLMSISE.h"

//Function to properly structure NRLMSISE00 input and run it
//Inputs are: year, DOY, seconds, geographic latitude, longitude, altitude, iloc, F10.7, F10.7 smoothed 81, AP smoothed 8
//The outputs are He, O, N2, O2, Ar, Total mass, H, N, Anomalous oxygen, Exospheric temperature and Temperature at altitude
double *runNRLMSISE(double year, double doy, double sec, double lat, double lon, double alt, double iloc, double f10, double f10_81, double ap_8)
{
	//define variables
	struct nrlmsise_input msisInput;
    struct nrlmsise_output msisOutput;
    struct nrlmsise_flags flags;
	double *output;
	
	output = calloc(13, sizeof(double));
	
	//check for non-finite values
	if (ap_8 != ap_8) {ap_8 = 0;}
	
	//set input structure
	msisInput.doy = (long)doy;
    msisInput.year = (long)year;
    msisInput.sec = (double)sec;
    msisInput.alt = (double)alt;
    msisInput.g_lat = (double)lat;
    msisInput.g_long = (double)lon;
    msisInput.lst = (double)iloc;
    msisInput.f107A = (double)f10_81;
    msisInput.f107 = (double)f10;
    msisInput.ap = (double)ap_8;
	
	//set switches
	flags.switches[0] = 0;
	for (int i=1; i<24; i++) {flags.switches[i] = 1;}
	
	if (f10 == f10 && f10_81 == f10_81)
	{
	
		//run the model
		gtd7(&msisInput, &flags, &msisOutput);
		
		output[0] = msisOutput.d[0]; //He
		output[1] = msisOutput.d[1]; //O
		output[2] = msisOutput.d[2]; //N2
		output[3] = msisOutput.d[3]; //O2
		output[4] = msisOutput.d[4]; //ar
		output[5] = msisOutput.d[5]; //Total Mass
		output[6] = msisOutput.d[6]; //h
		output[7] = msisOutput.d[7]; //n
		output[8] = msisOutput.d[8]; //Anomolous Oxygen
		output[9] = msisOutput.t[0]; //Exospheric temp
		output[10] = msisOutput.t[1]; //temp at input altitude
		output[11] = output[0] + output[1] + output[2] + output[3] + output[4] + output[6] + output[7]; //total number
		output[12] = output[5] / (output[11] * 6.02252e23); //normalized mass
	}else
	{
		for (int i=0;i=12;i++){output[i] = 0.0/0.0;}
	}
	
	return output;
	
}