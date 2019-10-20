
find ./Z_focused_1/ -type f -name "12*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused_1/ -type f -name "37*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused_1/ -type f -name "38*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused_1/ -type f -name "03*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused_1/ -type f -name "04*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused_1/ -type f -name "05*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused_1/ -type f -name "06*" | xargs -i cp {} ./data_train/Healthy &

find ./Z_focused_2/ -type f -name "12*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused_2/ -type f -name "37*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused_2/ -type f -name "38*" | xargs -i cp {} ./data_train/Cancer &
find ./Z_focused_2/ -type f -name "03*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused_2/ -type f -name "04*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused_2/ -type f -name "05*" | xargs -i cp {} ./data_train/Healthy &
find ./Z_focused_2/ -type f -name "06*" | xargs -i cp {} ./data_train/Healthy &


wait
python3 randomfilerenamer.py ./data_train/Cancer/
python3 randomfilerenamer.py ./data_train/Healthy/
# python3 extractG.py