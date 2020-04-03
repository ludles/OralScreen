# command to run: 
# nohup ./collect_results.sh &

mkdir ./logs/

# Dataset 1 & 2
for dataset in 1 2; do
	for fold in 1 2 3; do
		for pre in 0 1; do
			for i in 1 2 3; do
				CUDA_VISIBLE_DEVICES=1 python main.py -d $dataset -a ResNet50 -p $pre -f $fold -i $i -m train -s 1 >logs/data${dataset}_ResNet50_pre${pre}_fold${fold}_${i}.out 2>&1
				CUDA_VISIBLE_DEVICES=1 python main.py -d $dataset -a DenseNet201 -p $pre -f $fold -i $i -m train -s 1 >logs/data${dataset}_DenseNet201_pre${pre}_fold${fold}_${i}.out 2>&1
			done
		done
	done
done

# Dataset 3
for fold in 1 2; do
	for pre in 0 1; do
		for i in 1 2 3; do
			CUDA_VISIBLE_DEVICES=2 python main.py -d 3 -a ResNet50 -p $pre -f $fold -i $i -m train -s 1 >logs/data3_ResNet50_pre${pre}_fold${fold}_${i}.out 2>&1
			CUDA_VISIBLE_DEVICES=2 python main.py -d 3 -a DenseNet201 -p $pre -f $fold -i $i -m train -s 1 >logs/data3_DenseNet201_pre${pre}_fold${fold}_${i}.out 2>&1
		done
	done
done