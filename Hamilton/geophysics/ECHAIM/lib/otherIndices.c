#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "otherIndices.h"
#include "date.h"
#include "mathlib.h"
#include "global.h"

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

#define M_PI 3.14159265358979323846


//Get the solar zenith angle and local time
//input DOY, hr, geographic lat and lon
double * zenSun(double doy, double hr, double lat, double lon)
{
	static double r[2]; //return values
	
	double nday[] = {
		-4.0,  1.0,   6.0,  11.0,  16.0,  21.0,  26.0,  31.0,  36.0,  41.0,  46.0,
		51.0,  56.0,  61.0, 66.0,  71.0,  76.0,  81.0,  86.0,  91.0,  96.0,
		101.0, 106.0, 111.0, 116.0, 121.0, 126.0, 131.0, 136.0, 141.0, 146.0,
		151.0, 156.0, 161.0, 166.0, 171.0, 176.0, 181.0, 186.0, 191.0, 196.0,
		201.0, 206.0, 211.0, 216.0, 221.0, 226.0, 231.0, 236.0, 241.0, 246.0,
		251.0, 256.0, 261.0, 266.0, 271.0, 276.0, 281.0, 286.0, 291.0, 296.0,
		301.0, 306.0, 311.0, 316.0, 321.0, 326.0, 331.0, 336.0, 341.0, 346.0,
		351.0, 356.0, 361.0, 366.0, 371.0
	};

	double eqt[] = {
		-0.62, -3.23, -5.49, -7.60, -9.48, -11.09, -12.39, -13.34, -13.95, -14.23, -14.19,
		-13.85, -13.22, -12.35, -11.26, -10.01, -8.64, -7.18, -5.67, -4.16, -2.69,
		-1.29, -0.02, 1.10, 2.05, 2.80, 3.33, 3.63, 3.68, 3.49, 3.09,
		2.48, 1.71, 0.79, -0.24, -1.33, -2.41, -3.45, -4.39, -5.20, -5.84,
		-6.28, -6.49, -6.44, -6.15, -5.60, -4.82, -3.81, -2.60, -1.19, 0.36,
		2.03, 3.76, 5.54, 7.31, 9.04, 10.69, 12.20, 13.53, 14.65, 15.52,
		16.12, 16.41, 16.36, 15.95, 15.19, 14.09, 12.67, 10.93, 8.93, 6.70,
		4.32, 1.86, -0.62, -3.23, -5.49
	};

	double dec[] = {
		-23.35, -23.06, -22.57, -21.91, -21.06, -20.05, -18.88, -17.57, -16.13, -14.57, -12.91,
		-11.16, -9.34, -7.46, -5.54, -3.59, -1.62, 0.36, 2.33, 4.28, 6.19,
		8.06, 9.88, 11.62, 13.29, 14.87, 16.34, 17.70, 18.94, 20.04, 21.00,
		21.81, 22.47, 22.95, 23.28, 23.43, 23.40, 23.21, 22.85, 22.32, 21.63,
		20.79, 19.80, 18.67, 17.42, 16.05, 14.57, 13.00, 11.33, 9.60, 7.80,
		5.95, 4.06, 2.13, 0.19, -1.75, -3.69, -5.62, -7.51, -9.36, -11.16,
		-12.88, -14.53, -16.07, -17.50, -18.81, -19.98, -20.99, -21.85, -22.52, -23.02,
		-23.33, -23.44, -23.35, -23.06, -22.57
	};
	
	double dd = fmod(doy + (hr/24) - 1, 365.25) + 1;
	
	//get the length of the nday array
	int n = sizeof(nday)/sizeof(double);
	
	//double *eqtime = spline(inpute, x, 5, xout, 1);
	double *eqtime = spline(eqt, nday, n, &dd, 1);
	eqtime[0] /= 60.0;
	double *latsun = spline(dec, nday, n, &dd, 1);
	double lonsun = (-15.0) * (hr - 12 + eqtime[0]);

	//compue the solar zenith azimuth and flux mutliplier
	double t0 = (90.0 - lat) * (M_PI / 180.0);
	double t1 = (90 - latsun[0]) * (M_PI / 180.0);
	double p0 = lon * (M_PI / 180.0);
	double p1 = lonsun * (M_PI / 180.0);
	double zz = cos(t0) * cos(t1) + sin(t0) * sin(t1) * cos(p1 - p0);
	//double xx = sin(t1) * sin(p1 - p0);
	//double yy = sin(t0) * cos(t1) - cos(t0) * sin(t1) * cos(p1 - p0);
	//double azimuth = ((-1) * atan2(yy,xx)) * (180.0 / M_PI);
	double zenith = acos(zz) * (180.0 / M_PI);
	
	//double rsun = 1.0 - 0.01673 * cos((0.9856 * (dd - 2)) * (M_PI / 180.0));
	//double solfac = zz / (rsun * rsun);
	
	double local_time = (lon - lonsun)/15 + 12;
	
	if (local_time > 24) local_time -= 24;
	if (local_time < 0) local_time += 24;
	
	r[0] = zenith;
	r[1] = local_time;
	
	//free memory
	free(eqtime);
	free(latsun);
	
	return r;
	
}

//calculate the dipole tilt
//takes a julian date as input
double dpTilt (double jd)
{
	//See Matlab or IDL code for more robust comments
		
	//variables
	int k;
	double f1, f2, g10, g11, h11, dt;
	
	//set the Gregorian date variables
	int *g = gregDate(jd);
	int yr=2000; //year is hardcoded until a refit is done
	int mon=g[1], da=g[2], hr=g[3], min=g[4], s=g[5];
	
	free(g);
		
	//calculate the DOY
	int doy = (int)(julianDate(yr,mon,da,0,0,0) - julianDate(yr,1,1,0,0,0) + 1);
	
	//check for the 1900 to 2020 interval
	if (yr < 1900 || yr > igrf_maxyear + 5)
	{
		printf("DIPTILT-IGRFCOEFS:Year must be between 1900 and %i\n",igrf_maxyear+5);
		return 0;
	}
		
	if (yr > igrf_maxyear) {yr = igrf_maxyear+5;}

	//Getting components corresponding to the input year
	if (yr <= igrf_maxyear)
	{
		k = ((yr - 1900) / 5);
		f2 = (yr + ((doy - 1) / 365.25) - (((k-1) * 5) + 1900)) / 5;
		f1 = 1 - f2;
		g10 = (f1 * igrf_g0[k]) + (f2 * igrf_g0[k+1]);
		g11 = (f1 * igrf_g1[k]) + (f2 * igrf_g1[k+1]);
		h11 = (f1 * igrf_h[k]) + (f2 * igrf_h[k+1]);
		
		
	}else
	{
		dt = yr + ((doy - 1) / 365.25) - igrf_maxyear;
		g10 = igrf_g0[igrf_l0-1] - igrf_g0ex*dt;
		g11 = igrf_g1[igrf_l0-1] + igrf_g1ex*dt;
		h11 = igrf_h[igrf_l0-1] + igrf_hex*dt; 
	}

	//calcuate the unit vector EzMag components in geo coord system
	double sq = (g11*g11) + (h11*h11);
	double sqq = sqrt(sq);
	double sqr = sqrt((g10*g10) + sq);
	double sl0 = -h11/sqq;
	double cl0 = -g11/sqq;
	double st0 = sqq/sqr;
	double ct0 = g10/sqr;
	double stcl = st0*cl0;
	double stsl = st0*sl0;

	//[srasn,sdec,gst,slong]
	double *r = sun(yr, doy, hr, min, s);
	
	double ss[3] = {cos(r[0]) * cos(r[1]), sin(r[0]) * cos(r[1]), sin(r[1])};
	double cgst = cos(r[2]);
	double sgst = sin(r[2]);
	
	double dip[3] = {(stcl * cgst) - (stsl * sgst), (stcl * sgst) + (stsl * cgst), ct0};
	double sps = (dip[0] * ss[0]) + (dip[1] * ss[1]) + (dip[2] * ss[2]);
	double psi = asin(sps);

	return psi;

}

//sun position parameters
double * sun (int yr, int doy, int hr, int min, int s)
{
	//See Matlab or IDL code for more robust comments
	
	double fday = ((hr * 3600.0) + (min * 60.0) + s) / 86400.0;
	double dj = 365 * (yr - 1900) + (int)((yr - 1901)/4) + doy - 0.5 + fday;
	double t = dj / 36525;
	double v1 = fmod((279.696678 + 0.9856473354*dj), 360.0);
	double gst = fmod((279.690983 + .9856473354 * dj + 360 * fday + 180.), 360.0);
	gst = gst * (M_PI / 180.0);
	double g = fmod((358.475845 + 0.985600267 * dj), 360.0);
	g = g * (M_PI / 180.0);
	double slong = (v1 + (1.91946 - 0.004789 * t) * sin(g) + 0.020094 * sin(2 * g));
	slong = slong * (M_PI / 180.0);

	if (slong > (2 * M_PI)) slong -= (2 * M_PI);
	else if (slong < 0) slong += (2 * M_PI);
	
	double obliq = (23.45229 - 0.0130125 * t) * (M_PI / 180.0);
	double sob = sin(obliq);
	double slp = slong - 0.00009924;
	
	double sind = sob * sin(slp);
	double cosd = sqrt(1 - (sind * sind));
	double sc = sind / cosd;
	double sdec = atan(sc);
	double srasn = M_PI - atan2((cos(obliq) / sob) * sc, (-1) * cos(slp) / cosd);
	
	static double r[4];
	r[0] = srasn;
	r[1] = sdec;
	r[2] = gst;
	r[3] = slong;
	
	return r;
}

