#ifndef CALCHMF1_H
#define CALCHMF1_H

#include "sqlite3.h"

double *calcHmF1 (double *jd, double *glat, double *glon, double *lat, double *mlt, int l0, sqlite3 *db, sqlite3 *dbCoefs);

#endif
