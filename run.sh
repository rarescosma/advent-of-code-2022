#!/usr/bin/env bash

DAY=13

for x in $(seq -w 1 $DAY); do 
    echo ">>> Day $x <<<"
    if test -f $x.py
    then 
        python $x.py 
    elif test -x target/release/day$x
    then 
        ./target/release/day$x 2>/dev/null
    fi
    echo 
done

