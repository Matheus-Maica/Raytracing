#ifndef CALCMF2_H
#define CALCMF2_H

#include "sqlite3.h"

double **calcMF2 (double *jd, double *glat, double *glon, double *lat, double *lon, int l0, int option, sqlite3 *db, sqlite3 *dbCoefs);

#endif
