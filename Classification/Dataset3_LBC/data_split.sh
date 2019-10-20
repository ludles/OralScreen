mkdir data_train
mkdir data_train/Cancer
mkdir data_train/Healthy
mkdir data_test
mkdir data_test/Cancer
mkdir data_test/Healthy
mkdir logs

find ./Z_focused/ -type f -name "01*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused/ -type f -name "05*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused/ -type f -name "53*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused/ -type f -name "63*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused/ -type f -name "59*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused/ -type f -name "80*" | xargs -i cp {} ./data_train/Healthy &

find ./Z_focused/ -type f -name "07*" | xargs -i cp {} ./data_test/Cancer &
find ./Z_focused/ -type f -name "37*" | xargs -i cp {} ./data_test/Cancer &
find ./Z_focused/ -type f -name "55*" | xargs -i cp {} ./data_test/Cancer &
find ./Z_focused/ -type f -name "26*" | xargs -i cp {} ./data_test/Healthy &
find ./Z_focused/ -type f -name "61*" | xargs -i cp {} ./data_test/Healthy &
find ./Z_focused/ -type f -name "78*" | xargs -i cp {} ./data_test/Healthy &

wait
cp -r ./data_train/ ./data_train_byslide/
python3 randomfilerenamer.py ./data_train/Cancer/
python3 randomfilerenamer.py ./data_train/Healthy/
# python3 extractG.py