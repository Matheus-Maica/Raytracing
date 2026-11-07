#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mathlib.h"
#include <omp.h>

//****DEBUGGING************
#ifdef DBG
#include "memwatch.h"
#endif
//*************************

//Math functions

//function to boxcar smooth an array (input) or length (l)
//the function uses a boxcar of width (w)
double * smooth(double *input, int l, int w)
{
	//variables
	double *output; //output array
	output = calloc(l, sizeof(double)); //allocate the memory
	double total=0; //temporary total
	int ti, nc=0; //temporary index, nan counter
	
	if ((w % 2) == 0) { w++;}

	//iterate through the array
	for (int i=0; i<l; i++)
	{
		
		//check for the index being out of range
		//if it is then return the raw value for this point
		if (i < (w/2))
		{
			output[i] = input[i];
			continue;
		}
		if (i > (l - 1) - (w/2))
		{
			output[i] = input[i];
			continue;
		}
	
		//iterate through the window
		for (int j=0; j<w; j++)
		{
			//set the temporary index
			ti = i+j - (w/2);
						
			//checking if the value is a nan
			if (input[ti] != input[ti])
			{
				nc++;
				continue;
			}
			
			//add to total
			total += input[ti];
		}
		
		//if there were nans then use IDL technique
		if (nc)
		{
			//add the average of all valid points for each nan
			total += (total / (w - nc)) * nc;
				
		}
		
		//set the output value
		output[i] = total/w;
		
		//set total and nan counter back to zero
		total = 0;
		nc = 0;
		
	}
	
	return output;
	
}

//calculates the legendre polynomials and evaluates at x*Pn
//returns value for degree n and order m
//code based on matlba source code legendre.m
double legendre(double x, int n, int m)
{
	//if x>1 then p is undefined
	if (fabs(x)>1) return NAN;
	
	//if |m|>n then Px=0
	if (abs(m) > n) return 0.0;
	
	//if m<0, polynomials are proportional to those of m>0
	//calculate the proportionality constant
	double cfnm=1;
	if (m<0)
	{
		m = (-1.0) * m;
		
		cfnm = (pow(-1.0,m) * fact(n-m)) / fact(n+m);
	}
	
	//calculate coefficient of maximum degree in x*Pn
	//using explicit analytical expression
	double cl = (pow(-1.0,m) * cfnm * fact(2*n)) / (pow(2.0,n) * fact(n) * fact(n-m));
	double px = n - m;
	
	//setting x^2
	double x2 = x * x;
	
	//calculate polynomial part of expression
	double p = cl;

	for (int i=n-1; i>=0; i--)
	{
		//if the exponent of x is <= 2 break
		if (px<2) break;

		//calculate polynomial
		cl = ((((-1.0) * (i + i + 2.0 - n - m)) * (i + i + 1.0 - n - m)) / (2.0 * (i + i + 1.0) * (n - i))) * cl;

		//add the polynomial
		p = (p * x2) + cl;
		
		//decrease the exponent
		px = px - 2;
	}
	
	//if the exponent of the polynomial is 1 (odd)
	//multiply by x
	if (px==1) p *= x;

	//if m==0 then done
	if (m==0) return p;
	
	//if m != 0 then multiply by following factor
	x2 = 1.0 - x2;
	
	for (int i=1; i<=floor(m/2.0); i++)
	{
		p *= x2;
	
	}
	
	//if m is odd there is an additional factor
	if (m % 2) p *= sqrt(x2);

	return p;
}

// Binary search function to find the index of the largest x[j] <= xout
int findIndex(double *x, int l1, double xout_val) {
    int low = 0, high = l1 - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (x[mid] == xout_val)
            return mid;
        if (x[mid] < xout_val)
            low = mid + 1;
        else
            high = mid - 1;
    }
    return low - 1;  // Return the index where xout_val fits
}

//interpolate routine using linear interpolation
//input = input y values
//x = corresponding x values
//l1 = length of input/xout
//xout = the points to interpolate to
//l2 = length of xout
//assumes length of input and x are the same
double* interp(double *input, double *x, int l1, double *xout, int l2) {
    double *output = calloc(l2, sizeof(double));  // Allocate memory for output

    for (int i = 0; i < l2; i++) {
        // Handle out-of-bound cases
        if (xout[i] < x[0] || xout[i] > x[l1 - 1]) {
            output[i] = NAN;
            continue;
        }

        // Find the index in x using binary search
        int j = findIndex(x, l1, xout[i]);

        // Handle exact match
        if (xout[i] == x[j]) {
            output[i] = input[j];
        } else {
            // Perform linear interpolation
            double x1 = x[j];
            double x2 = x[j + 1];
            double y1 = input[j];
            double y2 = input[j + 1];
            output[i] = y1 + (y2 - y1) * (xout[i] - x1) / (x2 - x1);
        }
    }

    return output;
}

//Cubic spline interpolation
//input = input y values
//x = corresponding x values
//l1 = length of input/xout
//xout = the points to interpolate to
//l2 = length of xout
//assumes length of input and x are the same
//Nans will propogate through the entire calculation
//they must be dealt with before using the spline routine
// Numerical Analysis 9th ed - Burden, Faires (Ch. 3 Natural Cubic Spline, Pg. 149)
double * spline (double *input, double *x, int l1, double *xout, int l2)
{
	//declare variables
	double *h, *l, *u, *z, *c, *b, *d, *alpha;
	double *output, dx=0;
	int j;
	
	//initializing arrays
	h = calloc(l1, sizeof(double));
    l = calloc(l1,sizeof(double));
	u = calloc(l1,sizeof(double));
	z = calloc(l1,sizeof(double));
	c = calloc(l1,sizeof(double));
	b = calloc(l1,sizeof(double));
	d = calloc(l1,sizeof(double));
	alpha = calloc(l1-1,sizeof(double));
	output = calloc(l2, sizeof(double));
	
	//calculate the h values
    for (int i=0; i<=l1-2; i++)
	{
		h[i] = x[i+1] - x[i];
	}

    //calculate alpha values
    for (int i=1; i<=l1-2; i++)
	{
        alpha[i] = (3.0 * (input[i+1] - input[i]) / h[i]) - (3.0 * (input[i] - input[i-1]) / h[i-1]);
	}
	
	//initiate the first value
	l[0] = 1;
	u[0] = 0;
    z[0] = 0;

    
    for (int i=1; i<=l1-2; i++)
	{
        l[i] = (2.0 * (x[i+1] - x[i-1])) - (h[i-1] * u[i-1]); //x[i+1] - x[i-1]
        u[i] = h[i] / l[i];
        z[i] = (alpha[i] - (h[i-1] * z[i-1])) / l[i];
    }

    //set final values
    l[l1-1] = 1;
    z[l1-1] = 0;
    c[l1-1] = 0;

    // Step 6 
    for (int j = l1-2; j >= 0; j--) 
	{
        c[j] = z[j] - (u[j] * c[j+1]);
        b[j] = ((input[j+1] - input[j]) / h[j]) - ((h[j] * (c[j+1] + (2.0 * c[j]))) / 3.0);
        d[j] = (c[j+1] - c[j]) / (3.0 * h[j]);
    }
	
	
	//looping through each value in xout
	for (int i=0; i<l2; i++)
	{
		//loop through the x array to figure out where the xout value lies
		for (j=1; j<l1; j++)
		{			
			
			if ((xout[i] > x[j-1]) && (xout[i] < x[j])) break;
		}
		
		//set the output value for the corresponding xout
		dx = xout[i] - x[j-1];
		output[i] = input[j-1] + (b[j-1] * dx) + (c[j-1] * (dx * dx)) + (d[j-1] * (dx * dx * dx));
	}
	
	//checking if any values are identical to the input x values
	for (int i=0; i<=l2-1; i++)
	{
		for (int j=0; j<=l1-1; j++)
		{
			if (xout[i] == x[j])
				output[i] = input[j];
		}
	}
	
	
	//free memory
	free(alpha);
	free(h);
	free(l);
	free(u);
	free(z);
	free(c);
	free(d);
	free(b);
	
	return output;
   
}

//routine to get the factorial of a number
double fact(int in)
{
	double out = 1;
	
	for (int i=2; i<=in; i++)
	{
		out *= i;
	}
	
	return out;
}

//routine to give the index associated with the
//minimum value in the provided array
//length of l0
int minInd(double *array, int l0)
{
    int r = 0;
    #pragma omp parallel
    {
        int local_r = 0;
        #pragma omp for
        for (int i = 1; i < l0; i++)
        {
            if (array[i] < array[local_r]) local_r = i;
        }

        #pragma omp critical
        {
            if (array[local_r] < array[r]) r = local_r;
        }
    }
    return r;
}


//routine to give the index associated with the
//maximum value in the provided array
//length of l0
int maxInd(double *array, int l0)
{
	//output index/index of currently lowest value
	int r=0;
	
	//loop through the array
	for (int i=1; i<l0; i++)
	{
		//if current value is lower, set r
		if(array[i] > array[r]) r=i; 
	}
	
	return r;
}

//cotes 5 sides numerical integration
//returns the numerical integration
//inx = independent variable (must be evenly spaced)
//f = dependent variable
//l0 = length of the input arrays
double cotes(double *inx, double *f, int l0)
{
	//declare variables
	double output, h, endpts, *x, *g;
	int c = 0, n;
	
	n = l0;
	
	while ((n % 4) != 0) {c++;}
	
	//assign intermediate values
	x = calloc(l0+c+1, sizeof(double));
	
	//set new length
	n = n + c + 1;
	
	//populate x
	h = (inx[l0-1] - inx[0]) / (n - 1);
	x[0] = inx[0];
	for (int i=1; i<n; i++) 
	{
		x[i] = x[i-1] + h;;
	}
	
	//interpolate f to these new values
	g = spline(f, inx, l0, x, n);
	
	endpts = g[0] + g[n-1];
	
	double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
	for (int i=0; i<n; i+=4)
	{
		if (1+i < l0) {sum1 += g[1+i];}
		if (2+i < l0) {sum2 += g[2+i];}
		if (3+i < l0) {sum3 += g[3+i];}
		if (4+i < l0) {sum4 += g[4+i];}
	}
	
	//calculate the integral output
	output = (4*h/90) * (7*endpts+32*sum1 + 12*sum2 + 32*sum3 + 14*sum4);
	
	//free allocated memory
	free(x);
	free(g);
	
	//return output value
	return output;
}
