#ifndef CALCPRECIP_H
#define CALCPRECIP_H

#include "sqlite3.h"

double **calcPrecip (double *jd, double *lat, double *lon, int l0, double *alt, int l1, sqlite3 *db);

#endif