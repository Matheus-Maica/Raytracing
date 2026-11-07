#include <stdlib.h>
#include "FIRITools.h"

//functions used in the D-region FIRI modules

//This function creates the 5D index from a 1D index
//specific to the FIRI arrays
#include <stdlib.h>

int* indexTo5D(int input)
{
    // Define the sizes of each dimension
    int indices[5] = {3, 14, 11, 3, 101};
    
    // Allocate memory for the output
    int *output = calloc(5, sizeof(int));
    
    // Directly compute the indices
    output[0] = input % indices[0];
    output[1] = (input / indices[0]) % indices[1];
    output[2] = (input / (indices[0] * indices[1])) % indices[2];
    output[3] = (input / (indices[0] * indices[1] * indices[2])) % indices[3];
    output[4] = (input / (indices[0] * indices[1] * indices[2] * indices[3])) % indices[4];
    
    return output;
}


//This function takes the 1D FIRI arrays and returns the 5D array
double *****arrayTo5D(double *input)
{
	//define variables
	double *****output; 
	int indices[5] = {3,14,11,3,101};
	int cnt = 0;
	
	//define output memory
	output = calloc(indices[0], sizeof(double));
	for (int a=0; a<indices[0]; a++)
	{
		output[a] = calloc(indices[1], sizeof(double));
		for (int b=0; b<indices[1]; b++)
		{
			output[a][b] = calloc(indices[2], sizeof(double));
			for (int c=0; c<indices[2]; c++)
			{
				output[a][b][c] = calloc(indices[3], sizeof(double));
				for (int d=0; d<indices[3]; d++)
				{
					output[a][b][c][d] = calloc(indices[4], sizeof(double));
				}
			}
		}
	}
	
	for (int a=0; a<indices[4]; a++)
	{
		for (int b=0; b<indices[3]; b++)
		{
			for (int c=0; c<indices[2]; c++)
			{
				for (int d=0; d<indices[1]; d++)
				{
					for (int e=0; e<indices[0]; e++)
					{
						output[e][d][c][b][a] = input[cnt];
						cnt++;
					}
				}
			}
		}
	}
	
	return output;
	
}