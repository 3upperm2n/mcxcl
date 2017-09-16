#!/bin/sh

../../bin/mcxcl -A 3 -f benchmark1.json -k ../../src/mcx_core.cl -b 0 "$@"
