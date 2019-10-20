# -*- coding: utf-8 -*-
from tqdm import tqdm
import csv
import shutil
import matlab.engine
import os
import numpy as np
import sys

z_offsets = ['z-2000', 'z-1600', 'z-1200', 'z-800', 'z-400', 'z0', 
             'z400', 'z800', 'z1200', 'z1600', 'z2000']
patch_dir = './Patches/Z_expanded/'
best_dir = './Patches/Z_focused/'

#%%
def calculate_scores_vanilla(eng, img_name, i):
    
    scores = []
    for z_offset in z_offsets:
        patch_name ="{}_{}_{}.jpg".format(img_name, '0'*(4 - len(str(i))) + str(i), z_offset)
        patch_path = patch_dir + patch_name
        # read image through matlab
        patch = eng.imread(patch_path)
        # calcute quality score of the patch by EMBM
        Q_image = eng.EdgeModelBasedIQA(patch)
        scores.append(Q_image)

    return scores

#%%
def calculate_scores(eng, img_name, i, deviation=2):
    
    patches = []
    for z_offset in z_offsets:
        patch_name ="{}_{}_{}.jpg".format(img_name, '0'*(4 - len(str(i))) + str(i), z_offset)
        patch_path = patch_dir + patch_name
        # read image through matlab
        try:
            patch = eng.imread(patch_path)
        except:
            print("Could not open:", patch_path)
            return [0]
        
        
        # Pre-processing
        # MATLAB median filter
        patch = eng.medfilt3(patch, matlab.int8([5, 5, 1]))
        patches.append(patch)
        
    stds = []   # create a std list for the difference of patch focuses 
    for j in range(len(z_offsets) - 1):
        diff = np.asarray(patches[j + 1]) - np.asarray(patches[j])
        stds.append(np.std(diff))
        
    i_diff_peak = stds.index(max(stds)) # index of the difference peak
    i_start = max(0, i_diff_peak - deviation)
    i_stop = min(len(z_offsets) - 1, i_diff_peak + 1 + deviation)
    
    scores = [0] * len(patches)
    for j in range(len(patches)):
        if i_start <= j and j <= i_stop: # measure Q_image only near diff peak
            scores[j] = eng.EdgeModelBasedIQA(patches[j])

    return scores

#%%
def select_focus(quality_threshold=0.03):
    eng = matlab.engine.start_matlab()

    # read split paths from file
    with open('./debugging/temp.csv', 'r', newline='') as f:
        csv_paths = [line for line in csv.reader(f)]
    
    pid = int(sys.argv[1]) - 1
    csv_paths_needed = csv_paths[pid]
# =============================================================================
#     # the first chunk is used and deleted
#     csv_paths_needed = csv_paths.pop(0)
#     
#     # renew the file
#     with open('./debugging/temp.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         for line in csv_paths:
#             writer.writerow(line)
# =============================================================================

    for csv_path in tqdm(csv_paths_needed):
        img_name = os.path.basename(csv_path)[:-len('.csv')]
        with open(csv_path) as f:
            f_csv = csv.reader(f)
            patch_total = len(list(f_csv)) - 1
        patch_count = 0
        
        # for each patch position in the image
        for i in range(patch_total):
            scores = calculate_scores(eng, img_name, i)
                
            i_best = scores.index(max(scores)) # index of the best score
            name_best = "{}_{}_{}.jpg".format(img_name, '0'*(4 - len(str(i))) + str(i), z_offsets[i_best])
            # filter out the totally blured ones out of focus patches
            if max(scores) > quality_threshold:
                shutil.copyfile(patch_dir + name_best, best_dir + name_best)
                patch_count += 1
                
        print("{}/{} patches selected for {}".format(patch_count, patch_total, img_name))

    eng.quit()
    return

#%%
if __name__ == '__main__':
    print('#'*30)
    print('Selecting the most focused patches.')
    print('#'*30)    
    select_focus()