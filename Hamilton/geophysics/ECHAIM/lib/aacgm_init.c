#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include "aacgm_init.h"

int aacgm_init(const char *path) {
    DIR *dir;
    struct dirent *entry;
    char igrf_coeffs[512] = {0};
    char igrf_coeffs_short[512] = {0};
    char aacgm_prefix[512] = {0};
    char modified_path[512] = {0};


    // Append "AACGM_coeffs" to the input path
    snprintf(modified_path, sizeof(modified_path), "%s%sAACGM_coeffs", path, (path[strlen(path) - 1] == '/') ? "" : "/");

    if (!path) {
        fprintf(stderr, "Error: ECHAIMConfig path is NULL.\n");
        return -1;
    }

    // Open the directory
    dir = opendir(modified_path);
    if (!dir) {
        perror("Error opening AACGM_coeffs directory");
        return -1;
    }

    // Search for files matching the patterns
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "magmodel", 8) == 0) {
            if (snprintf(igrf_coeffs, sizeof(igrf_coeffs), "%s%s%s", modified_path, (modified_path[strlen(modified_path) - 1] == '/') ? "" : "/", entry->d_name) >= sizeof(igrf_coeffs)) {
                fprintf(stderr, "Error: IGRF_COEFFS path is too long.\n");
                closedir(dir);
                return -1;
            }
        } else if (strncmp(entry->d_name, "igrf", 4) == 0) {
            if (snprintf(igrf_coeffs_short, sizeof(igrf_coeffs_short), "%s%s%s", modified_path, (modified_path[strlen(modified_path) - 1] == '/') ? "" : "/", entry->d_name) >= sizeof(igrf_coeffs)) {
                fprintf(stderr, "Error: IGRF_COEFFS_SHORT path is too long.\n");
                closedir(dir);
                return -1;
            }
        } else if (strncmp(entry->d_name, "aacgm_coeffs", 12) == 0) {
            // Extract the prefix part only up to the second dash
            char *first_dash = strchr(entry->d_name, '-');
            if (first_dash) {
                char *second_dash = strchr(first_dash + 1, '-');
                if (second_dash) {
                    size_t prefix_len = second_dash - entry->d_name + 1; // Include up to the second dash
                    if (snprintf(aacgm_prefix, sizeof(aacgm_prefix), "%s%s%.*s", modified_path, (modified_path[strlen(modified_path) - 1] == '/') ? "" : "/", (int)prefix_len, entry->d_name) >= sizeof(aacgm_prefix)) {
                        fprintf(stderr, "Error: AACGM_v2_DAT_PREFIX path is too long.\n");
                        closedir(dir);
                        return -1;
                    }
                }
            }
        }

        // Exit early if both files are found
        if (igrf_coeffs[0] && aacgm_prefix[0] && igrf_coeffs_short[0]) {
            break;
        }
    }

    closedir(dir);

    // Check if all files were found
    if (!igrf_coeffs[0]) {
        fprintf(stderr, "Error: magmodel file not found in %s.\n", path);
        return -1;
    }

    if (!igrf_coeffs_short[0]) {
        fprintf(stderr, "Error: igrf file not found in %s.\n", path);
        return -1;
    }

    if (!aacgm_prefix[0]) {
        fprintf(stderr, "Error: aacgm_coeffs file not found in %s.\n", path);
        return -1;
    }

    // Set environment variables
    if (setenv("IGRF_COEFFS", igrf_coeffs, 1) != 0) {
        perror("Error setting IGRF_COEFFS");
        return -1;
    }

    if (setenv("IGRF_COEFFS_SHORT", igrf_coeffs_short, 1) != 0) {
        perror("Error setting IGRF_COEFFS_SHORT");
        return -1;
    }

    if (setenv("AACGM_v2_DAT_PREFIX", aacgm_prefix, 1) != 0) {
        perror("Error setting AACGM_v2_DAT_PREFIX");
        return -1;
    }

    return 0;
}
