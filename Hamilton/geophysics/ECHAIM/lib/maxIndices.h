#include "sqlite3.h"

#ifndef MAXINDICES_H
#define MAXINDICES_H

double maxAE(int mm,sqlite3 *db);
double maxDST(int mm,sqlite3 *db);
double maxKP(int mm,sqlite3 *db);
double maxAP(int mm,sqlite3 *db);
double maxPC(int mm,sqlite3 *db);
double maxF10(int mm,sqlite3 *db);
double maxF10F(int mm,sqlite3 *db);
double maxIG(int mm,sqlite3 *db);
double maxIG12(int mm,sqlite3 *db);

#endif