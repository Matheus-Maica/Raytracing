#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "densityFunc.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//echaim_bottomside_function.m
//Takes altitudes, hmF2, hmF1, hmE, HO, hF1, HE as inputs
//all inputs are scalar
//Routines scale factor(?) used in creating density profiles from NmF2
double botFunc(double alt,double hmf2,double hmf1,double hme,double ho,double hf1,double he)
{
	//declare variables
	//output, differences placeholders
	double output, altf2, f1f2, ef2;
	double v0, v1, v2, v3;
	
	altf2 = alt - hmf2;
	f1f2 = hmf1 - hmf2;
	ef2 = hme - hmf2;
	
	//check if f1 peak height is greater than f2 peak height
	if (hmf1 > hmf2) {hf1 = 0;}
	
	v0 = (altf2 - (ef2 - 15.0)) / 2.5;
	v1 = 1.0 / (1.0 + exp((-1.0) * v0));
	v2 = hf1 / pow(cosh((altf2 - f1f2) / fabs(f1f2 / 2.5)),2.0);
	v3 = he / pow(cosh((altf2 - ef2) / 30.0),2.0);
	
	//if these values are less than zero then set to zero
	if (ho < 0 || ho != ho) {ho = 0;}
	if (v2 < 0 || v2 != v2) {v2 = 0;}
	if (v3 < 0 || v3 != v3) {v3 = 0;}
	
	//set scale height
	double sh = (ho + v2 + v3) * v1;
	
	if (sh < 0) {sh = 0;}
	
	output = pow(cosh(altf2 / sh), -2.0);
	
	return output;
	
	
}

//NeQuick_model_new_r_g.m
double topFunc (double alt, double hmf2, double htop)
{
	//declare variables
	double output, v0;
	
	v0 = htop * (1.0 + ((20.0 * 0.18 * (alt - hmf2)) / (20.0 * htop + 0.18 * (alt - hmf2))));
	output = 4.0 * exp((alt - hmf2) / v0) / pow(1 + exp((alt - hmf2) / v0),2.0);
	
	return output;
}