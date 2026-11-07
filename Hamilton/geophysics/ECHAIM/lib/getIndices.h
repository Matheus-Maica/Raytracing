#ifndef GETINDICES_H
#define GETINDICES_H

#include "sqlite3.h"

double *AE (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *AEraw (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *DST (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *KP (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *AP (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *AP_8 (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *APraw (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *F10 (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *F10_27 (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *F10_81 (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *IG (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *PC (double jd0, double jd1, double **x, int **l0, sqlite3 *db);
double *PCraw (double jd0, double jd1, double **x, int **l0, sqlite3 *db);

#endif