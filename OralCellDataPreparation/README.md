# Oral Cell Data Preparation

## Dependencies

- [NDPITools](https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/) (choose NDPITools software version, independent of ImageJ)
- [Install MATLAB Engine API for Python](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html?lang=en) 
- Other packages for python

## Usage

1. Clone this repository to the directory of the NDPI files.

2. There are two options for this step:

    1. Edit the path for `inputFolder` and `outputFolder` in `particle_analysis_fixedDirectory.ijm` : 
        ```
        inputFolder="/PATH_TO/OralCellDataPreparation/PredMasks/";
        outputFolder="/PATH_TO/OralCellDataPreparation/CSVResults/";
        ```
        
    2. Or alternatively, replace 
    
        ```bash
        ImageJ --headless -macro particle_analysis_fixedDirectory.ijm
        ```
        
        with
        
        ```bash
        ImageJ -macro particle_analysis.ijm
        ```
        in `data_preparation.sh`. Then the path for `inputFolder` and `outputFolder` will need to be selected when prompted.

3. Run

   ```bash
   cd OralCellDataPreparation
   ./data_preparation.sh
   ```
   
   The focused patches will be generated in `/PATH_TO/OralCellDataPreparation/Patches/Z_focused/`.
   
   Or replace 
   
   ```bash
   nohup python3 select_focus.py >./nohup.out 2>./nohup.err &
   ```
   
   with
   
   ```bash
   ./select_focus_parallel.sh N
   ```
   
   to run in parallel, where input argument N is the number of processes.
   
   The progress bar is redirected in `/PATH_TO/OralCellDataPreparation/debugging/*.err`, together with the error messages.

## Known Problems

- There might be some "imread error". This is due to trying to read some corrupted images. They are handled by being ignored.

