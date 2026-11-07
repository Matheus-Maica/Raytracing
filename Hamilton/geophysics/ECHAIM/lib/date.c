#include <math.h>
#include <stdlib.h>
#include "date.h"

#include <stdio.h>

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Converting from Gregorian to Julian date
double julianDate (int y, int m, int d, int h, int min, int sec)
{
	//https://quasar.as.utexas.edu/BillInfo/JulianDatesG.html
	//https://idlastro.gsfc.nasa.gov/ftp/pro/astro/juldate.pro
	
	//if month is 1 or 2, don't include leap day
	if (m < 3)
	{
		--y;
		m += 12;
	}
	
	int a = floorf(y/100);
	double jd = floorf(y * 0.25) + (365.0 * (y - 1860.0)) \
		+ floorf(30.6001 * (m + 1)) + d - 105.5;
		
	if (jd > -100830.5) jd = jd + 2 - a + floorf(a*0.25);

	//Adding the hours, minutes, and seconds
	jd += (h/24.0) + (min/1440.) + (sec/86400.0);

	return jd + 2400000.0;
	
}

//convert from Julian date to Gregorian
int * gregDate (double jd)
{
	//declare variables
	static int *r; //output array
	int m, y; //month and year
	
	r = calloc(6, sizeof(int));

	double q = jd + 0.5;
	int z = (int)(q);
	int w = (int)((z - 1867216.25)/36524.25);
	int x = (int)(w / 4.0);
	int a = z + 1.0 + w - x;
	int b = a + 1524.0;
	int c = (int)((b - 122.1) / 365.25);
	int d = (int)(365.25 * c);
	int e = (int)((b - d) / 30.6001);
	int f = (int)(30.6001 * e);
	
	//day
	int day = b - d - f + (q - z);
	
	//month
	if (e <= 13) m = (e - 1); else m = e - 13;
	
	//year
	if (m <= 2) y = c - 4715; else y = c - 4716;
	
	//calculating the hour
	double jj = q - (int)(q);
	int hr = floorf(jj * 24.0);

	//Calculating minutes
	int min = round((jj * 1440.0) - (hr * 60.0));
	
	//Calculating seconds
	int sec = round((jj * 86400.0) - (hr * 3600.0) - (min * 60.0));
	
	r[0] = y; //year
	r[1] = m; //month
	r[2] = day; //day
	r[3] = hr; //hours
	r[4] = min; //minutes
	r[5] = sec; //seconds
	
	return r;
}

//Floors the given JD to the first of the month
double jdMonthly (double jd)
{
	int *r = gregDate(jd);
	double jo = julianDate(r[0], r[1], 1, 0, 0, 0);
	
	free(r);
	
	return jo;
}

//Floors the given JD to the first hour of the day
double jdDaily (double jd)
{
	int *r = gregDate(jd);
	
	double jo = julianDate(r[0], r[1], r[2], 0, 0, 0);
	
	free(r);
	
	return jo;
}

//Floors the given JD to the appropiate three hourly
double jdThreeHour (double jd)
{
	int *r = gregDate(jd);
	double jo = julianDate(r[0], r[1], r[2], r[3] - (r[3] % 3), 0, 0);
	
	free(r);
	
	return jo;
}

//get the doy
double jdDOY (double jd)
{
	int *r = gregDate(jd);
	double jo = (jdDaily(jd) - julianDate(r[0], 1, 1, 0, 0, 0)) + 1;
	
	free(r);
	
	return jo;
}

//Round the JD to the start of the hour
double jdHourly (double jd)
{
	int *r = gregDate(jd);
	double jo = julianDate(r[0], r[1], r[2], r[3], 0, 0);
	
	free(r);
	
	return jo;
}
