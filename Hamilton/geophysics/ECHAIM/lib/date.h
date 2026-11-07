#ifndef DATE_H
#define DATE_H

double julianDate (int y, int m, int d, int h, int min, int sec);
int * gregDate (double jd);
double jdMonthly (double jd);
double jdDaily (double jd);
double jdThreeHour (double jd);
double jdDOY(double jd);
double jdHourly (double jd);

#endif