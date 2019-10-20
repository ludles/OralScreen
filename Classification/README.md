# Classification



Files in the root directory apply to all datasets, but adjustment to specific dataset may be required.

- `ResNet50_aug.py` – main script to train the model with 1/8 random augmentation.
- `evaluate_slide_csv_std.py` – to evaluate the performance on each slide for all folds **3 times**. The **average** results will be written to a `fold_slide_results_data{1-3}.csv` file, last column being the **STD**. (3 trained models are required for each dataset.)



In each dataset, the `ResNet_aug{1-3}.hdf5` in each `./logs{1-{2|3}}/` represent 3 models trained for 2 or 3 folds. 



Here are some common files exists in all 3 datasets' directory.

- `data_split.sh` – to split data from `../Patches/Z_focused/` into  `../Patches/data_train{1-3}/` and `../Patches/data_test{1-3}/`, should be adjusted to each dataset before using.
- `randomfilerenamer.py` – rename files in a directory randomly. This should be used on the training data before training. It is called in the `data_split.sh`.



## Dataset 1: old smear data

Dataset 1 is relatively small. It can be trained and evaluated fast on local laptops. Here contains the code to use locally and also for some attempts:

- `./slide_mosaic1_oldsmear/` – mosaics consists of 10 * 10 patches from each glass, in order to give an overview of the dataset.

## Dataset 2: new smear data

Dataset 2 is specifically used for evaluating the impact of 

- focus level
- color channel

on the classification results. Here contains several files:

- `focus_augmentation.py` – to generate out-of-focus patches in separate folders like `./Z_focused_n/`.
- `data_split_focusaug_train.sh` – to copy data at several focus levels into training folder, and rename the data randomly by calling `randomfilerenamer.py`.
- `data_split_focusaug_test.sh` – to split data in `./Z_focused_n/`s into different testing folders like `./FocusAugEvaluation/data_test_n/`.
- `focusaug_evaluate.py` – to evaluate the performance of a trained model on **test sets at different focus levels** and write the results into `./focus_results_data2fold1.csv` .
- `extractG.py` – to exact **Green** channel of the data, and save the original RGB data to backup folders.
- `./slide_mosaic2_newsmear/` – mosaics consists of 16 * 16 patches from each glass, in order to give an overview of the dataset.

## Dataset 3: LBC data

- `./slide_mosaic3_LBC/` – mosaics consists of 16 * 16 patches from each glass, in order to give an overview of the dataset.

## Others

Here contains other files for development.

- `evaluate_slide.py` – to evaluate the performance on each slide. The results will be printed in the terminal.
- `evaluate_slide_csv.py` – to evaluate the performance on each slide **for all folds**. The results will be written to a `fold_slide_results_data{1-3}.csv` file. (Dataset 3 contains only 2 folds.)

  

  


