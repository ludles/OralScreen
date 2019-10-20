mkdir data_train
mkdir data_train/Cancer
mkdir data_train/Healthy
mkdir data_test
mkdir data_test/Cancer
mkdir data_test/Healthy
mkdir logs

find ./Z_focused/ -type f -name "12*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused/ -type f -name "37*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused/ -type f -name "38*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused/ -type f -name "03*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused/ -type f -name "04*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused/ -type f -name "05*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused/ -type f -name "06*" | xargs -i cp {} ./data_train/Healthy &

find ./Z_focused/ -type f -name "36*" | xargs -i cp {} ./data_test/Cancer &
find ./Z_focused/ -type f -name "07*" | xargs -i cp {} ./data_test/Healthy &
find ./Z_focused/ -type f -name "08*" | xargs -i cp {} ./data_test/Healthy &

wait
cp -r ./data_train/ ./data_train_byslide/
python3 randomfilerenamer.py ./data_train/Cancer/
python3 randomfilerenamer.py ./data_train/Healthy/
# python3 extractG.py