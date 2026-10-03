#ifndef ECHAIM_H
#define ECHAIM_H

double * NmF2(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err);

double * NmF2Storm(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err);

double * HmF2(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err);
				
double * HmF1(double *lat, double *lon, double *year, double *month, double *day, \
				double *hour, double *min, double *sec, int l0, int err);

double * densityPath(double *lat, double *lon, double *alt, \
						double *year, double *month, double *day, \
						double *hour, double *min, double *sec, int storm, int precip, int dregion, int l0, int err);
						
double ** densityProfile(double *lat, double *lon, double *year, double *month, double *day, \
						double *hour, double *min, double *sec, int storm, int precip, int dregion, int l0, double *alt, int l1, int err);
						
int updateLocalDB (int force);

double getDBDate();
				
#endif
