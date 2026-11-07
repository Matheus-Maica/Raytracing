#ifndef AACGM_INIT_H
#define AACGM_INIT_H

#ifdef _WIN32
#include <windows.h>

int setenv(const char *name, const char *value, int overwrite) {
    if (overwrite || GetEnvironmentVariable(name, NULL, 0) == 0) {
        return SetEnvironmentVariable(name, value) ? 0 : -1;
    }
    return 0;
}

int unsetenv(const char *name) {
    return SetEnvironmentVariable(name, NULL) ? 0 : -1;
}
#endif

int aacgm_init(const char *path);

#endif
