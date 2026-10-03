#ifndef GETPRECIPMODEL_H
#define GETPRECIPMODEL_H

#include "sqlite3.h"

double *getPrecipAEMean(sqlite3 *db, int isae);
double **getPrecipMLT(sqlite3 *db);
double **getPrecipMLAT(sqlite3 *db);
double *getPrecipEnergy(int row, int col, int tab, sqlite3 *db, int isae);
double *getPrecipFlux(int row, int col, int tab, sqlite3 *db, int isae);

#endif