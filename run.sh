#!/usr/bin/env bash

DAY=25
INTERPRETER="${1:-python}"

for x in $(seq -w 1 $DAY); do
    echo ">>> Day $x <<<"
    if test -f day$x.py
    then
        $INTERPRETER day$x.py
    elif test -x target/release/day$x
    then
        ./target/release/day$x 2>/dev/null
    fi
    echo
done

