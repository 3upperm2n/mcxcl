#!/bin/sh

../../bin/mcxcl -T 64 -f ben2a.json -b 1 -P '{"Shapes":[{"Sphere":   {"Tag":2, "O":[30,30,30],"R":15}}]}' -s benchmark2 -k ../../src/mcx_core.cl -J "-DUSE_ATOMIC" "$@"
