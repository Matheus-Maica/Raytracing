#ifndef _MATHLIB_H_
#define _MATHLIB_H_

double * smooth(double *input, int l, int w);
double legendre(double x, int n, int m);
double * interp (double *input, double *x, int l1, double *xout, int l2);
double * spline (double *input, double *x, int l1, double *xout, int l2);
double fact(int in);
int minInd(double *array, int l0);
int maxInd(double *array, int l0);
double cotes(double *inx, double *f, int l0);

#endif
