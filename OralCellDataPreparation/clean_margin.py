# -*- coding: utf-8 -*-

import os
from glob import glob

JPEGPATH = "./"
# %%
def clean_margin():
    img_names = [os.path.basename(p) for p in glob(JPEGPATH + '*z0*.jpg')]
    
    ndpis = set([img_name[:2] for img_name in img_names])
    
    for ndpi in ndpis:
        sub_imgs = [os.path.basename(p)[:-len('.jpg')] for p in glob(JPEGPATH + ndpi + '*z0*.jpg')]
        i_max = max([sub_img[sub_img.rindex('i'):sub_img.rindex('j')] for sub_img in sub_imgs])
        j_max = max([sub_img[sub_img.rindex('j'):] for sub_img in sub_imgs])
        margins = set(glob(JPEGPATH + ndpi + '*' + i_max + '*.jpg') + glob(JPEGPATH + ndpi + '*' + j_max + '*.jpg'))
        for margin in margins:
            os.remove(margin)
        print("{} margin images with {} or {} deleted for slide {}".format(len(margins), i_max, j_max, ndpi))

if __name__ == '__main__':
    clean_margin()