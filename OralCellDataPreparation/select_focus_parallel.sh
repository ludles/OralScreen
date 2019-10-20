#!/bin/bash

n=$1

# split the work to n chunks
python3 parallel_split.py $n

for i in `seq 1 $n`
do
	errname="${i}.err"
	outname="${i}.out"
	# echo "Number: $errname $outname"
	nohup python3 select_focus_parallel.py ${i} >./debugging/$outname 2>./debugging/$errname &
done