mkdir ./FocusAugEvaluation


for f in {0..10}
do
	datadir="./Z_focused_${f}/"
	testdir="./FocusAugEvaluation/data_test_f${f}"
	testdir0="./FocusAugEvaluation/data_test_f${f}/Cancer"
	testdir1="./FocusAugEvaluation/data_test_f${f}/Healthy"
	mkdir $testdir
	mkdir $testdir0
	mkdir $testdir1
	find $datadir -type f -name "36*" | xargs -i cp {} $testdir0 &
	find $datadir -type f -name "07*" | xargs -i cp {} $testdir1 &
	find $datadir -type f -name "08*" | xargs -i cp {} $testdir1 &
	# echo $datadir
	# echo $testdir
	# echo $testdir0
	# echo $testdir1
done




