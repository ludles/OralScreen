# Focus selection module
## Usage

- `FocusSelectionEval.py` – to evaluate the performance of the FS module **using EMBM with improvements**. (Default parameters are tuned to achieve the best performance on our dataset).
- `FocusSelectionEval_trystd.py` – to evaluate the performance of the FS module **using our method only**. There is 0.04 performance drop but about 600 speedup. (Default parameters are tuned to achieve the best performance on our dataset).
- `EdgeModelBasedIQA.m`, `edge_width.m`, and `mag2.cpp` are from [EMBM](https://github.com/GUAN3737/EMBM).

## References

- [No-reference blur assessment based on edge modeling](https://linkinghub.elsevier.com/retrieve/pii/S1047320315000085) (code:[EMBM](https://github.com/GUAN3737/EMBM)) 





##### Copyright from EMBM:

All Rights Reserved. This copyright statement may not be removed from any file containing it or from modifications to these files. This copyright notice must also be included in any file or product that is derived from the source files.

Redistribution and use of this code in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistribution's of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistribution's in binary form must reproduce the above copyright  notice, this list of conditions and the following disclaimer in    the  documentation and/or other materials provided with the distribution.

The code and our papers are to be cited in the bibliography as:

Jingwei Guan, Wei Zhang, Jason Gu and Hongliang Ren. "No-reference  blur assessment based on edge modeling." J. Vis. Commun. Image  Represent. 29 (2015) 1¨C7.

DISCLAIMER: This software is provided by the copyright holders and contributors "as  is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a  particular purpose are disclaimed. In no event shall Shandong  University, members, authors, or contributors be liable for any direct,  indirect, incidental, special, exemplary, or consequential damages  (including, but not limited to, procurement of substitute goods or  services; loss of use, data, or profits; or business interruption)  however caused and on any theory of liability, whether in contract,  strict liability, or tort (including negligence or otherwise) arising in  any way out of the use of this software, even if advised of the  possibility of such damage.

