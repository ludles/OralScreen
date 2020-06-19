[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](./LICENSE.md) [![](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat)](https://www.python.org/download/releases/3.6.0/) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 

# Oral Cell Screening Project

Code of paper [A Deep Learning based Pipeline for Efficient Oral Cancer Screening on Whole Slide Images](https://doi-org.ezproxy.its.uu.se/10.1007/978-3-030-50516-5_22)

- [Pre-print version on arXiv](http://arxiv.org/abs/1910.10549)

------

## Usage instruction

Each sub-directory contains a separate README instruction. File paths in the code might need to be changed before running.

- [`OralCellDataPreparation/`](./OralCellDataPreparation/) – Nucleus Detection (ND) and Focus Selection (FS) modules trained on our data. This will prepare all nucleus patches for classification. It can be tuned to better performance on a new dataset by the code in two directories below:
  - [`NucleusDetection/`](./NucleusDetection/) – to customise the ND module. 
  - [`FocusSelection/`](./FocusSelection/) – to customise the FS module. 
- [`Classification/`](./Classification/) – Classification module.

## Example results

<div align="center">
    <img src="./img/OC2_mosaic_03.jpg" width="40%"> <img src="./img/OC3_mosaic_37.jpg" width="40%">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Left: pineline on pap-smear data; Right: pineline on liquid-based data</div>
</div>

## Dependencies

[`oralscreen_env.yml`](./oralscreen_env.yml) includes the **full** list of packages used to run the experiments. Some packages might be unnecessary.

## Citation

Please cite our paper if you find the code useful for your research.

- J. Lu *et al.*, “A Deep Learning based Pipeline for Efficient Oral Cancer Screening on Whole Slide Images,” *International Conference on  Image Analysis and Recognition*, 2020, LNCS, vol 12132.

```
@inproceedings{OralScreen,
  title = {A {{Deep Learning Based Pipeline}} for {{Efficient Oral Cancer Screening}} on {{Whole Slide Images}}},
  booktitle = {Image {{Analysis}} and {{Recognition}}},
  author = {Lu, Jiahao and Sladoje, Nataša and Runow Stark, Christina and Darai Ramqvist, Eva and Hirsch, Jan-Michaél and Lindblad, Joakim},
  date = {2020},
  pages = {249--261},
  publisher = {{Springer International Publishing}},
  location = {{Cham}},
  doi = {10.1007/978-3-030-50516-5_22},
  isbn = {978-3-030-50516-5},
  langid = {english},
  series = {Lecture {{Notes}} in {{Computer Science}}}
}
```

## Acknowledgement

This work is supported by: Swedish Research Council proj. 2015-05878 and 2017-04385, VINNOVA grant 2017-02447, and FTV Stockholms Län AB.