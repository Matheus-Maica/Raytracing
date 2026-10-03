#include <math.h>
#include <stdlib.h>
#include "mathlib.h"

//Input the mean energy value, returns energy and flux spectrum (1000 elements)
double **gauss_energy_spectrum(double ener_mean)
{
	//define variables
	double er[2] = {log10(0.01),log10(400.0)};
	long n_elem = 10;
	int n_seg = 10;
	int alpha = 1;
	double **output; //[n,0] = energy; [n,1] = flux
	double temp[n_elem];
	
	//set gaussLeg values
	double x[10] = {-0.973906528517172, -0.865063366688985, -0.679409568299024, -0.433395394129247, -0.148874338981631, \
		0.148874338981631, 0.433395394129247, 0.679409568299024, 0.865063366688985, 0.973906528517172};
	double w[10] = {0.066671344308688, 0.149451349150580, 0.219086362515982, 0.269266719309996, 0.295524224714753, \
		0.295524224714753, 0.269266719309996, 0.219086362515982, 0.149451349150580, 0.066671344308688};
	
	//allocate memory for output
	output = calloc(3, sizeof(double *));
	for (int i=0; i<3; i++)
	{
		output[i] = calloc(n_seg*(n_seg-1), sizeof(double));
	}
	
	//create energy range
	double erange[n_elem];
	for (int i=0; i<n_seg; i++)
	{
		erange[i] = pow(10.0,(double) i / (n_seg-1) * (er[1] - er[0]) + er[0]);
	}
	
	//energy and flux calc
	double fluxTotal = 0.0;
	for (int i=0; i<n_seg-1; i++)
	{
		for (int j=0; j<n_seg; j++)
		{
			//energy
			output[0][(i*n_seg)+j] = (((x[j]+1)/2.0) * (erange[i+1] - erange[i])) + erange[i];
			
			//flux pt 1
			output[1][(i*n_seg)+j] = pow(output[0][(i*n_seg)+j],alpha) * exp((-1.0*output[0][(i*n_seg)+j]) / ener_mean);
			output[2][(i*n_seg)+j] = 0.5 * w[j] * (erange[i+1] - erange[i]);
			fluxTotal += output[1][(i*n_seg)+j] * output[2][(i*n_seg)+j];
		}
	}
	
	//normalize flux
	for (int i=0; i<(n_seg*(n_seg-1)); i++)
	{
			output[1][i] = output[1][i] / fluxTotal;
	}
	
	//return output array
	return output;
}

//Input geographic latitude, energy and flux arrays (see energy_sepctrum for lengths)
//altitude and atmData (output from runNRLMSISE)
//returns e and ion rates, length = length of energy/flux
double *eion_rate(double lat, double *energy, double *flux, double alt, double *atmData)
{
	//set variables
	long l0 = 90;  //length of energy flux, see energy_spectrum for length
	double *output;
	double c[8], ifunc = 0, pij = 0, y = 0; //intermediate variables
	
	double D2R = 0.017453292519943295769;
	double k = 1.38064852e-23;
	double g = 9.807;
	
	double welmec_g = (1 + 0.0053024 * pow(sin(lat * D2R), 2) - 0.0000058 * sin(2 * lat * D2R)) * 9.780318 - 0.000003085 * alt * 1000.0;
	
	double atm_mass = atmData[5] / atmData[11] / 1000;
	double scale_height = k * atmData[10] / atm_mass / welmec_g * 100;
	
	//allocate memory for output
	output = calloc(l0, sizeof(double));
	
	//set array
	double pij_tab[4][8] = {{1.24616, 2.23976, 1.41754, 2.48775e-1, -4.65119e-1, 3.86019e-1, -6.45454e-1, 9.48930e-1},\
					{1.45903, -4.22918E-7, 1.44597E-1, -1.50890E-1, -1.05081E-1, 1.75430E-3, 8.49555E-4, 1.97385E-1},\
					{-2.42269e-1, 1.36458E-2, 1.70433E-2, 6.30894E-9, -8.95701E-2, -7.42960E-4, -4.28581E-2, -2.50660E-3},\
					{5.95459e-2, 2.53332E-3, 6.39717E-4, 1.23707E-3, 1.22450E-2, 4.60881E-4, -2.99302E-3, -2.06938E-3}};
			
	//loop through elements in energy
	for (int el=0; el<l0; el++)
	{			
		for (int i=0; i<8; i++)
		{
			pij = 0;
			for (int j=0; j<4; j++)
			{
				pij += pij_tab[j][i] * pow(log(energy[el]),j);
			}
			c[i] = exp(pij);
		}
		
		y = 2.0 / energy[el] * pow(atmData[5] * scale_height / 6e-6,0.7);
		ifunc = (c[0] * pow(y,c[1])) * exp(-1 * c[2] * (pow(y,c[3]))) \
				+ (c[4] * pow(y,c[5])) * exp(-1 * c[6] * pow(y,c[7]));
		output[el] = ifunc * flux[el] / 35e-3 / scale_height;
	}
	
	return output;
	
}