#ifndef CALCPERT_H
#define CALCPERT_H

#include "sqlite3.h"

double *calcPert (double *jd, double *lat, double *lon, int l0, sqlite3 *db, sqlite3 *dbCoefs);

#endif
