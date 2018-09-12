#!/bin/bash
set -e
fname=$1
target=$2
eval_step=10000
mkdir -p $target
for i in {0..2}
do
	mkdir -p $fname/run_$i
	python run_condet.py -l $fname/run_$i -e $eval_step -s $i
done
cp -r $fname $target