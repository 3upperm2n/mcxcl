#!/bin/sh
if [ ! -e semi60x60x60.bin ]; then
  dd if=/dev/zero of=semi60x60x60.bin bs=1000 count=216
  perl -pi -e 's/\x0/\x1/g' semi60x60x60.bin
fi

#1m
../../bin/mcxcl -t 16384 -T 64 -g 10 -n 1e6 -f qtest.inp -s qtest -r 1 -a 0 -b 0 -k ../../src/mcx_core.cl -d 0
#5m
../../bin/mcxcl -t 16384 -T 64 -g 10 -n 5e6 -f qtest.inp -s qtest -r 1 -a 0 -b 0 -k ../../src/mcx_core.cl -d 0
