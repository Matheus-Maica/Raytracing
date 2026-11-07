#!/bin/bash
# Builds libechaim.so from the vendored E-CHAIM C sources in lib/.
#
# updateLocalDB.c is deliberately excluded: it's the only file requiring
# libcurl (a system -dev package not installed here), and it only serves
# the optional "auto-refresh CHAIM_DB.db from chain-project.net" feature.
# Hamilton runs off the bundled etc/CHAIM_DB.db instead -- consistent with
# how igrf.py/ionosphere.py also rely on bundled/local coefficient data
# rather than a live network fetch. Refreshing CHAIM_DB.db for very recent
# dates is a manual step (see ECHAIM/README.md) until that's wired back in.
set -euo pipefail
cd "$(dirname "$0")"

SOURCES=$(ls lib/*.c | grep -v updateLocalDB.c)

echo "Compiling $(echo "$SOURCES" | wc -l) source files..."
gcc -fPIC -O2 -std=gnu17 -Ilib -c $SOURCES
gcc -shared -o libechaim.so *.o -lm -lpthread -ldl
rm -f ./*.o

echo "Built $(pwd)/libechaim.so"
