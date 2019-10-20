# Focus selection module
This part is developed partially based on the code of paper [No-reference blur assessment based on edge modeling](https://linkinghub.elsevier.com/retrieve/pii/S1047320315000085)(EMBM). 

- `FocusSelectionEval.py` – to evaluate the performance of the FS module **using EMBM with improvements**. (Default parameters are tuned to achieve the best performance on our dataset).
- `FocusSelectionEval_trystd.py` – to evaluate the performance of the FS module **using our method only**. There is 0.04 performance drop but about 600 speedup. (Default parameters are tuned to achieve the best performance on our dataset).
- `EdgeModelBasedIQA.m`, `edge_width.m`, and `mag2.cpp` are from [EMBM](https://github.com/GUAN3737/EMBM).


