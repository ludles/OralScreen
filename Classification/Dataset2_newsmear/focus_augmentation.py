# -*- coding: utf-8 -*-
import os
import shutil
from glob import glob
from tqdm import tqdm

# %%
def select_defocus_patches(n_defocus, mode='one'):
    """Generates out-of-focus patches in separate folders.

    Args:
        n_defocus: To select patches of what out-of-focus level. 0 is in-focus.
        mode: 
            'one' -- only select patches at n_defocus level.
            'all' -- select patches at all focus levels within n_defocus.
    """
    if n_defocus <= 0 or n_defocus > len(z_offsets):
        print("n_defocus must be a proper positive integer.")
        return
    
    for off_focus_level in range(n_defocus, 0, -1):
        
        if mode == 'one' and off_focus_level != n_defocus:
            return
        
        tar_dir = "./Z_focused_" + str(off_focus_level) + "/"
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)
            
        for patch_path in tqdm(glob(focused_dir+'*.jpg')):
            patch_name = os.path.basename(patch_path)
            general_name = patch_name[:patch_name.rindex('z')]
            patch_focus = patch_name[patch_name.rindex('z'):-len('.jpg')]
            i_focus = z_offsets.index(patch_focus)
            
            if 0 <= i_focus - off_focus_level:
                left_name = general_name + z_offsets[i_focus - off_focus_level] + '.jpg'
                if os.path.exists(expanded_dir+left_name):
                    shutil.copyfile(expanded_dir+left_name, tar_dir+left_name)
                    
            if i_focus + off_focus_level <= len(z_offsets) - 1:
                right_name = general_name + z_offsets[i_focus + off_focus_level] + '.jpg'
                if os.path.exists(expanded_dir+right_name):
                    shutil.copyfile(expanded_dir+right_name, tar_dir+right_name)
                    
# %%
if __name__ == '__main__':
    z_offsets = ['z-2000', 'z-1600', 'z-1200', 'z-800', 'z-400', 'z0', 
             'z400', 'z800', 'z1200', 'z1600', 'z2000']
    focused_dir = "./Z_focused/"
    expanded_dir = "./Z_expanded/"
    select_defocus_patches(n_defocus=2, mode='all')