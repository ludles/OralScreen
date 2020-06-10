# Nucleus detection module

## Usage

1. `train.py` – to train the detector.

2. `threshold_opt.py` – to generate predicted masks with one of the trained weights at a certain threshold.
   1. `trained_model.hdf5` – weights of the model trained on LBC slides to detect **all** nuclei.
   2. `trained_model_strictdata.hdf5` – weights of the model trained on LBC & Smear slides (with more annotated samples) to detect **only free-lying** nuclei.
3. `particle_analysis.ijm` – an ImageJ macro script to exact the centroids of blobs from generated binary masks and save to `Results/*.csv`.
4. `test.py` – to read `Results/*.csv` and evaluate performance at a certain threshold.

## Other files

- `draw_sample.py` – to draw detection markers on a generated mask.
- `PerformanceTest/` – the directory to evaluate performance as `threshold` changes (similar to the code above but more concise and with all needed data included).

## Reference

- [Microscopy cell counting and detection with fully convolutional regression networks](https://www.tandfonline.com/doi/full/10.1080/21681163.2016.1149104) (code: [cell_counting_v2](https://github.com/WeidiXie/cell_counting_v2#cell_counting_v2))