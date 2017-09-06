#!/bin/sh

../../bin/mcxcl -T 64 -n 1e7 -f qtest.inp -k ../../src/mcx_core.cl "$@"

