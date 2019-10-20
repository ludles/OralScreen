mkdir data_train3
mkdir data_train3/Cancer
mkdir data_train3/Healthy
mkdir data_test3
mkdir data_test3/Cancer
mkdir data_test3/Healthy
# mkdir logs

find ./BenchmarkData/ -type f -name "12*" | xargs -i cp {} ./data_train3/Cancer &
find ./BenchmarkData/ -type f -name "36*" | xargs -i cp {} ./data_train3/Cancer &
find ./BenchmarkData/ -type f -name "03*" | xargs -i cp {} ./data_train3/Healthy &
find ./BenchmarkData/ -type f -name "04*" | xargs -i cp {} ./data_train3/Healthy &
find ./BenchmarkData/ -type f -name "05*" | xargs -i cp {} ./data_train3/Healthy &
find ./BenchmarkData/ -type f -name "06*" | xargs -i cp {} ./data_train3/Healthy &

find ./BenchmarkData/ -type f -name "37*" | xargs -i cp {} ./data_test3/Cancer &
find ./BenchmarkData/ -type f -name "38*" | xargs -i cp {} ./data_test3/Cancer &
find ./BenchmarkData/ -type f -name "07*" | xargs -i cp {} ./data_test3/Healthy &
find ./BenchmarkData/ -type f -name "08*" | xargs -i cp {} ./data_test3/Healthy &

wait
python3 randomfilerenamer.py ./data_train3/Cancer
python3 randomfilerenamer.py ./data_train3/Healthy
