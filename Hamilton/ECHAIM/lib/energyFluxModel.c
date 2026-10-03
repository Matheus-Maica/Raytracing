#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "energyFluxModel.h"
#include "getPrecipModel.h"
#include "global.h"
#include "mathlib.h"
#include "sqlite3.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//function which models the energy and flux for given location/AE
//all inputs are double arrays
//l0 = length of other arrays
//output[0][n] = energy; output[1][n] = flux
double **energyFluxModel(double *mlts, double *mlats, double *AEs, double *PCs, int l0)
{
	//declare variables
	double **output, *AErefs, *PCrefs, *refs, **outmlts, **outmlats;
	char cwd[1024]; 
	sqlite3 *db;
	
	//array dimensions from DB
	int AEDBL0 = 23;
	int PCDBL0 = 26;
	int ML0 = 50;
	int ML1 = 100;
	
	//open model DB (AE)
	strcpy(cwd,DIR);
	strcat(cwd,"PRECIP_DB.db");
	int rc = sqlite3_open(cwd, &db);
	if (rc) {printf("Error: PRECIP_DB could not be opened\n");}
	
	//allocate memory
	output = calloc(2, sizeof(double));
	for (int i=0; i<2; i++) {output[i] = calloc(l0, sizeof(double));}
	
	//get AE mean array from DB
	AErefs = getPrecipAEMean(db, 1);
	PCrefs = getPrecipAEMean(db, 0);
	outmlts = getPrecipMLT(db);
	outmlats = getPrecipMLAT(db);
	
	for (int i=0; i<l0; i++)
	{
		//limit AE to 1300 upper bound
		if (AEs[i] > 1300.0) {AEs[i] = 1300.0;}
		if (PCs[i] > 13.0) {PCs[i] = 13.0;}
		if (PCs[i] < -7.0) {PCs[i] = -7.0;}
		
		//set the input index to AE or PC
		int isae = 1;
		int dbl0 = AEDBL0;
		double val = AEs[i];
		refs = AErefs;
		if (val != val) 
		{
			isae = 0;
			dbl0 = PCDBL0;
			val = PCs[i];
			refs = PCrefs;
		}
		
		//If PC is nan too, then set to the mean AE=117.7
		if (val != val) 
		{
			isae = 1;
			dbl0 = AEDBL0;
			val = 117.7;
			refs = AErefs;
		}
		
		
		//get closest AE reference points
		double aet[dbl0];
		for (int j=0; j<dbl0; j++) {aet[j] = fabs(refs[j] - val);}
		int m1 = minInd(aet, dbl0);
		double AE1 = refs[m1];
		aet[m1] = 1e6;
		int m2 = minInd(aet, dbl0);
		double AE2 = refs[m2];
		
		//need grid of closest points of input mlt and mlat
		double x1, x2, y1, y2;
		int xt1, xt2, yt1, yt2;
		
		//mlt (x) col
		double t[ML0];
		for (int j=0; j<ML0; j++) {t[j] = fabs(outmlts[j][0] - mlts[i]);}
		int ti = minInd(t, ML0);
		xt1 = ti;
		x1 = outmlts[xt1][0];
		t[ti] = 1e6;
		ti = minInd(t, ML0);
		xt2 = ti;
		x2 = outmlts[xt2][0];
		
		//mlat (y) row
		double tt[ML1];
		for (int j=0; j<ML1; j++) {tt[j] = fabs(outmlats[0][j] - mlats[i]);}
		ti = minInd(tt, ML1);
		yt1 = ti;
		y1 = outmlats[0][yt1];
		tt[ti] = 1e6;
		ti = minInd(tt, ML1);
		yt2 = ti;
		y2 = outmlats[0][yt2];
		
		//get output_gridded* for found points
		//x1,y1 table AE1 (m1); x1,y1, table AE2 (m2)
		double *e111 = getPrecipEnergy(yt1, xt1, m1, db, isae);
		double *e112 = getPrecipEnergy(yt1, xt1, m2, db, isae);
		double *f111 = getPrecipFlux(yt1, xt1, m1, db, isae);
		double *f112 = getPrecipFlux(yt1, xt1, m2, db, isae);
		//x1,y2 table AE1; x1,y2, table AE2
		double *e121 = getPrecipEnergy(yt2, xt1, m1, db, isae);
		double *e122 = getPrecipEnergy(yt2, xt1, m2, db, isae);
		double *f121 = getPrecipFlux(yt2, xt1, m1, db, isae);
		double *f122 = getPrecipFlux(yt2, xt1, m2, db, isae);
		//x2,y1 table AE1; x2,y1, table AE2
		double *e211 = getPrecipEnergy(yt1, xt2, m1, db, isae);
		double *e212 = getPrecipEnergy(yt1, xt2, m2, db, isae);
		double *f211 = getPrecipFlux(yt1, xt2, m1, db, isae);
		double *f212 = getPrecipFlux(yt1, xt2, m2, db, isae);
		//x2,y2 table AE1; x2,y2, table AE2
		double *e221 = getPrecipEnergy(yt2, xt2, m1, db, isae);
		double *e222 = getPrecipEnergy(yt2, xt2, m2, db, isae);
		double *f221 = getPrecipFlux(yt2, xt2, m1, db, isae);
		double *f222 = getPrecipFlux(yt2, xt2, m2, db, isae);
		
		//use these values to do bilinear interpolation
		//https://en.wikipedia.org/wiki/Bilinear_interpolation	
		double outEn1 = (1.0 / ((x2 - x1) * (y2 - y1))) * \
						(e111[0] * (x2 - mlts[i]) * (y2 - mlats[i]) + \
						e211[0] * (mlts[i] - x1) * (y2 - mlats[i]) + \
						e121[0] * (x2 - mlts[i]) * (mlats[i] - y1) + \
						e221[0] * (mlts[i] - x1) * (mlats[i] - y1));
		double outEn2 = (1.0 / ((x2 - x1) * (y2 - y1))) * \
						(e112[0] * (x2 - mlts[i]) * (y2 - mlats[i]) + \
						e212[0] * (mlts[i] - x1) * (y2 - mlats[i]) + \
						e122[0] * (x2 - mlts[i]) * (mlats[i] - y1) + \
						e222[0] * (mlts[i] - x1) * (mlats[i] - y1));
		double outFl1 = (1.0 / ((x2 - x1) * (y2 - y1))) * \
						(f111[0] * (x2 - mlts[i]) * (y2 - mlats[i]) + \
						f211[0] * (mlts[i] - x1) * (y2 - mlats[i]) + \
						f121[0] * (x2 - mlts[i]) * (mlats[i] - y1) + \
						f221[0] * (mlts[i] - x1) * (mlats[i] - y1));
		double outFl2 = (1.0 / ((x2 - x1) * (y2 - y1))) * \
						(f112[0] * (x2 - mlts[i]) * (y2 - mlats[i]) + \
						f212[0] * (mlts[i] - x1) * (y2 - mlats[i]) + \
						f122[0] * (x2 - mlts[i]) * (mlats[i] - y1) + \
						f222[0] * (mlts[i] - x1) * (mlats[i] - y1));
						
		//set nan or negative values to 0
		if (outEn1 != outEn1 || outEn1 < 0.0) {outEn1 = 0.0;}
		if (outEn2 != outEn2 || outEn2 < 0.0) {outEn2 = 0.0;}
		if (outFl1 != outFl1 || outFl1 < 0.0) {outFl1 = 0.0;}
		if (outFl2 != outFl2 || outFl2 < 0.0) {outFl2 = 0.0;}
		
		//interpolate betweeen upper and lower references
		output[0][i] = outEn1 + (val - AE1) * (outEn2 - outEn1) / (AE2 - AE1);
		output[1][i] = outFl1 + (val - AE1) * (outFl2 - outFl1) / (AE2 - AE1);
	
		free(e111);	free(e112);	free(e121);	free(e122);
		free(e211);	free(e212);	free(e221);	free(e222);
		free(f111);	free(f112);	free(f121);	free(f122);
		free(f211);	free(f212);	free(f221);	free(f222);
	
	}
	
	//free memory
	free(AErefs);
	free(PCrefs);
	for (int i=0; i<ML0; i++) 
	{
		free(outmlts[i]);
		free(outmlats[i]);
	}
	free(outmlts);
	free(outmlats);
	
	//close the DB
	sqlite3_close(db);
	
	return output;
	
}
