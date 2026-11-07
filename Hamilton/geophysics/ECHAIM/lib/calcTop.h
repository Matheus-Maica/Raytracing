#ifndef CALCTOP_H
#define CALCTOP_H

#include "sqlite3.h"

double *calcTop (double *jd, double *glat, double *glon, double *lat, double *mlt, int l0, sqlite3 *db, sqlite3 *dbCoefs);


#endif
