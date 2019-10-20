# Oral Cell Screening Project
Code of paper "A Deep Learning based Pipeline for Efficient Oral Cancer Screening on Whole Slide Images"

Each sub-directory contains a separate README instruction. File paths in the code might need to be changed before running.



------

- `OralCellDataPreparation/` – Nucleus Detection (ND) and Focus Selection (FS) modules trained on our data. This will prepare all nucleus patches for classification. It can be tuned to better performance on a new dataset by the code in two directories below:
  - `NucleusDetection/` – to customise the ND module. 
  - `FocusSelection/` – to customise the FS module. 
- `Classification/` – Classification module.





## References

[1] [cell_counting_v2](https://github.com/WeidiXie/cell_counting_v2#cell_counting_v2)

[2] [EMBM](https://github.com/GUAN3737/EMBM) 

##### Copyright from EMBM:

All Rights Reserved. This copyright statement may not be removed from any file containing it or from modifications to these files. This copyright notice must also be included in any file or product that is derived from the source files.

Redistribution and use of this code in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistribution's of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistribution's in binary form must reproduce the above copyright  notice, this list of conditions and the following disclaimer in    the  documentation and/or other materials provided with the distribution.

The code and our papers are to be cited in the bibliography as:

Jingwei Guan, Wei Zhang, Jason Gu and Hongliang Ren. "No-reference  blur assessment based on edge modeling." J. Vis. Commun. Image  Represent. 29 (2015) 1¨C7.

DISCLAIMER: This software is provided by the copyright holders and contributors "as  is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a  particular purpose are disclaimed. In no event shall Shandong  University, members, authors, or contributors be liable for any direct,  indirect, incidental, special, exemplary, or consequential damages  (including, but not limited to, procurement of substitute goods or  services; loss of use, data, or profits; or business interruption)  however caused and on any theory of liability, whether in contract,  strict liability, or tort (including negligence or otherwise) arising in  any way out of the use of this software, even if advised of the  possibility of such damage.

