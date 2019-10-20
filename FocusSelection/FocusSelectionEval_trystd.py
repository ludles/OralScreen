# -*- coding: utf-8 -*-
'''
Evaluate the Focus Selection (FS) performance only using Standard Deviation (SD).
'''
from glob import glob
import os, cv2
import numpy as np
from tqdm import tqdm
#from matplotlib.pyplot import imshow
#from scipy import ndimage
#import matlab.engine

z_offsets = ['z-2000', 'z-1600', 'z-1200', 'z-800', 'z-400', 'z0', 
             'z400', 'z800', 'z1200', 'z1600', 'z2000']

patch_dir = '../VOC2007_old/Patches/Z_expanded/'
test_dir = '../VOC2007_old/Patches/Z_testset/'
#%%
def create_groundtruth(delta_median=2, delta_range=0):
    annos = np.empty([100, 0], dtype=int)
    
    for anno_path in glob("../VOC2007_old/Patches/Z_eval/annotation_*.txt"):
        with open(anno_path) as f:
            anno = f.read()
        anno_line = np.asarray([int(i) for i in anno.split()]).reshape(100, 1)
        annos = np.concatenate((annos, anno_line), axis=-1)
    
    medians = np.median(annos, axis=-1)
#    mean = np.mean(annos, axis=-1)
#    var = np.var(annos, axis=-1)
#    ptp = np.ptp(annos, axis=-1)
    mins = np.min(annos, axis=-1)
    maxs = np.max(annos, axis=-1)
    
    # Calculate human performance
    tps_median_list = []
    tps_range_list = []
    for person in range((annos.shape[1])):
        tps_m = 0
        tps_r = 0
        _annos = np.delete(annos, person, axis=-1)
        _medians = np.median(_annos, axis=-1)
        _mins = np.min(_annos, axis=-1)
        _maxs = np.max(_annos, axis=-1)
        
        for i in range((_annos.shape[0])):
            if annos[i, person] >= _medians[i] - delta_median and annos[i, person] <= _medians[i] + delta_median:
                tps_m += 1
            if annos[i, person] >= _mins[i] - delta_range and annos[i, person] <= _maxs[i] + delta_range:
                tps_r += 1
                
        tps_median_list.append(tps_m)
        tps_range_list.append(tps_r)
        
    tp_median_hmavg = np.mean(tps_median_list)
    tp_range_hmavg = np.mean(tps_range_list)
    
    return mins, maxs, medians, tp_median_hmavg, tp_range_hmavg

   

gt_min, gt_max, gt_median, tp_median_hmavg, tp_range_hmavg = create_groundtruth()



#%%
def count_TP(m=3, delta_median=2, delta_range=0, showN=False):
    
#    eng = matlab.engine.start_matlab()
    tp_range = 0
    tp_median = 0

    for i in range(len(glob(test_dir + '*.jpg'))):
        test_name = os.path.basename(glob(test_dir + '*.jpg')[i]).split('_')
        patch_pos = test_name[0] + '_' + test_name[1]
                  
            
        patches = []    # create a list of focuses
        for z_offset in z_offsets:
            patch_name = "{}_{}.jpg".format(patch_pos, z_offset)
            patch_path = patch_dir + patch_name
            patch = cv2.imread(patch_path)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = cv2.medianBlur(patch, m)
            patch = np.int32(patch)
            patches.append(patch)
            
        stds = [np.std(patches[j + 1] - patches[j]) for
                j in range(len(z_offsets) - 1)]
            
        i_diff_peak = stds.index(max(stds)) # index of the difference peak

# =============================================================================
        if i_diff_peak == 0:
            i_best = i_diff_peak + 1   # peak at first
        elif i_diff_peak == len(stds) - 1:
            i_best = i_diff_peak       # peak at last
        elif stds[i_diff_peak - 1] > stds[i_diff_peak + 1]:
            i_best = i_diff_peak        # peak at left
        else:
            i_best = i_diff_peak + 1    # peak at right
            

        if i_best >= gt_min[i] - delta_range and i_best <= gt_max[i] + delta_range:
            tp_range += 1
        if i_best >= gt_median[i] - delta_median and i_best <= gt_median[i] + delta_median:
            tp_median += 1
        
#    eng.quit()
    
    return tp_range, tp_median#, Negs
#%%
print('m\tacc_m\tacc_r')
for m in [1, 3, 5, 7]:
    tp_range, tp_median = count_TP(m=m)
    accuracy_range = tp_range / 100
    accuracy_median = tp_median / 100
    print('{}\t{}\t{}'.format(m, accuracy_median, accuracy_range))
    
accuracy_median_hm = tp_median_hmavg / 100
accuracy_range_hm = tp_range_hmavg / 100


