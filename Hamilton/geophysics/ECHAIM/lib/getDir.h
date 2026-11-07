#ifndef GETDIR_H
#define GETDIR_H

int getDir();

// Non-interactive initializer for embedding ECHAIM as a library (e.g. from
// Python via ctypes): sets DIR directly and runs the AACGM/IGRF coefficient
// init that getDir() would otherwise trigger, without ever touching
// configECHAIM.dat or stdin. Every public ECHAIM function calls getDir()
// internally; once echaim_init() has set DIR, getDir() becomes a no-op
// (see getDir.c), so callers only need to invoke this once at startup.
int echaim_init(const char *dir);

#endif