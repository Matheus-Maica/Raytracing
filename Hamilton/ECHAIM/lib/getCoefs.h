#ifndef GETCOEFS_H
#define GETCOEFS_H

#include "sqlite3.h"

double *getH0(sqlite3 *db);
double *getHE(sqlite3 *db);
double *getHF1(sqlite3 *db);
double *getHMF1(sqlite3 *db);
double **getHMF2(sqlite3 *db);
double **getNMF2(sqlite3 *db);
double *getNe(sqlite3 *db);
double **getPERT(sqlite3 *db);
double *getAE(sqlite3 *db);

#endif