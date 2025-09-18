# Neural network enabled wide field-of-view imaging with hyperbolic metalenses
[Paper](https://doi.org/10.1515/nanoph-2025-0354) | [Data](https://doi.org/10.5281/zenodo.14746073)

[Joel Yeo](https://orcid.org/0000-0001-5160-7628), [Deepak K. Sharma](https://orcid.org/0000-0002-5733-3952), [Saurabh Srivastava](https://orcid.org/0000-0001-6420-1440), [Aihong Huang](https://orcid.org/0000-0003-4609-173X), [Emmanuel Lassalle](https://orcid.org/0000-0002-0098-5159), [Egor Khaidarov](https://orcid.org/0000-0002-0848-552X), Keng Heng Lai, [Yuan Hsing Fu](https://orcid.org/0000-0002-7691-0196), [N. Duane Loh](https://orcid.org/0000-0002-8886-510X), [Arseniy Kuznetsov](https://orcid.org/0000-0002-7622-8939), [Ramon Paniagua-Dominguez](https://orcid.org/0000-0001-7836-681X)

This repository presents a collection of notebooks to perform spatially-varying deblurring of hyperbolic metalens camera images using a Restormer network demonstrated in our paper. Three notebooks are made available:
1. Creation of simulated spatially-varying blurred image dataset.
2. Restormer training.
3. Restormer prediction (deblurring).

## Installation
To run the code, users may first install the required packages listed in the root directory via ```pip install -r requirements.txt```.
Users may download the dataset from [Zenodo]() and place the files in ```hyperbolic-imaging/data/```.
After which, users may simply run the notebooks.

## Citation
If you find our work useful in your research, please cite:
```
@article{Yeo2025,
title = {Neural network enabled wide field-of-view imaging with hyperbolic metalenses},
author = {Joel Yeo and Deepak K. Sharma and Saurabh Srivastava and Aihong Huang and Emmanuel Lassalle and Egor Khaidarov and Keng Heng Lai and Yuan Hsing Fu and N. Duane Loh and Arseniy I. Kuznetsov and Ramon Paniagua-Dominguez},
journal = {Nanophotonics},
doi = {doi:10.1515/nanoph-2025-0354},
year = {2025},
lastchecked = {2025-09-18}
}
```

## License
Our code is licensed under GNU GPLv3. By downloading the software, you agree to the terms of this License.
