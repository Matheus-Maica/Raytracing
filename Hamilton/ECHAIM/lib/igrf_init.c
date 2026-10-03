#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "global.h"
#include "igrf_init.h"

// Global variables
double *igrf_g0 = NULL, *igrf_g1 = NULL, *igrf_h = NULL; // Data arrays
double igrf_g0ex, igrf_g1ex, igrf_hex;                           // Last values
int igrf_l0;                                               // Length of arrays
int igrf_maxyear;                                                 // Extracted year

void igrf_init() {
    const char *file_path = getenv("IGRF_COEFFS_SHORT"); // Get file path from environment variable
    if (!file_path) {
        fprintf(stderr, "Environment variable 'IGRF_COEFFS_SHORT' not set.\n");
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(file_path, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[1024]; // Buffer for reading lines
    int g10_count = 0, g11_count = 0, h11_count = 0; // Data counters
    int year_found = 0;
    int temp_year[] = {0, 0};

    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "g/h n m", 7) == 0 && !year_found) {
            // Extract year from the "g/h n m" line dynamically
            char *token = strtok(line, " ");
            while (token != NULL) {
                temp_year[0] = temp_year[1];
                temp_year[1] = atoi(token);
                
                token = strtok(NULL, " ");
            }

            igrf_maxyear = temp_year[0];

        } else if (strncmp(line, "g  1  0", 7) == 0) {
            // Parse "g 1 0" line
            char *token = strtok(line + 8, " ");
            while (token != NULL) {
                igrf_g0 = realloc(igrf_g0, (g10_count + 1) * sizeof(double));
                igrf_g0[g10_count++] = atof(token);
                token = strtok(NULL, " ");
            }
        } else if (strncmp(line, "g  1  1", 7) == 0) {
            // Parse "g 1 1" line
            char *token = strtok(line + 8, " ");
            while (token != NULL) {
                if (atof(token) != 1 || atof(token) != -1) {
                    igrf_g1 = realloc(igrf_g1, (g11_count + 1) * sizeof(double));
                    igrf_g1[g11_count++] = atof(token);
                }
                token = strtok(NULL, " ");
            }
        } else if (strncmp(line, "h  1  1", 7) == 0) {
            // Parse "h 1 1" line
            char *token = strtok(line + 8, " ");
            while (token != NULL) {
                if (atof(token) != 1 || atof(token) != -1) {
                    igrf_h = realloc(igrf_h, (h11_count + 1) * sizeof(double));
                    igrf_h[h11_count++] = atof(token);
                }
                token = strtok(NULL, " ");
            }
        }
    }

    fclose(file);

    //set the max length
    igrf_l0 = g10_count - 1;

    //set the interpolation values
    igrf_g0ex = igrf_g0[igrf_l0];
    igrf_g1ex = igrf_g1[igrf_l0];
    igrf_hex = igrf_h[igrf_l0];

    //set the desired sign for the coefficients
    for (int i=0; i<igrf_l0; i++) {
        igrf_g0[i] = (-1)*igrf_g0[i];
    }
}
