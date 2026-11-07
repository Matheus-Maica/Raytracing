#ifndef ENERGYFUNCTIONS_H
#define ENERGYFUNCTIONS_H

double **gauss_energy_spectrum(double ener_mean);
double *eion_rate(double lat, double *energy, double *flux, double alt, double *atmData);

#endif
