#ifndef ECHAIMFIRI2018_H
#define ECHAIMFIRI2018_H

#include "sqlite3.h"

double **ECHAIMFIRI2018 (double *jd, double *lat, double *lon, int l0, double *alt, int l1, sqlite3 *db);

#endif
