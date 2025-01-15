# Neural network enabled wide field-of-view imaging with hyperbolic metalenses
[Paper]() | [Data]()

[Joel Yeo](https://orcid.org/0000-0001-5160-7628), Deepak K. Sharma, Saurabh Srivastava, Aihong Huang, Emmanuel Lassalle, Keng Heng Lai, Yuan Hsing Fu, [N. Duane Loh](https://orcid.org/0000-0002-8886-510X), [Arseniy Kuznetsov](https://orcid.org/0000-0002-7622-8939), [Ramon Paniagua-Dominguez](https://orcid.org/0000-0001-7836-681X)
This repository presents a collection of notebooks to perform spatially-varying deblurring of hyperbolic metalens camera images using a Restormer network demonstrated in our paper. Three notebooks are made available:
1. Creation of simulated spatially-varying blurred image dataset.
2. Restormer training.
3. Restormer prediction (deblurring).

## Installation
To run the code, users may first install the required packages listed in the root directory via ```pip install -r requirements.txt```.
Users may download the dataset from [Zenodo]() and place the files in ```hyperbolic-image/data/```.
After which, users may simply run the notebooks.

## Citation
If you find our work useful in your research, please cite:
```
@article{YEO2024113962,
title = {Ghostbuster: A phase retrieval diffraction tomography algorithm for cryo-EM},
journal = {Ultramicroscopy},
volume = {262},
pages = {113962},
year = {2024},
issn = {0304-3991},
doi = {https://doi.org/10.1016/j.ultramic.2024.113962},
url = {https://www.sciencedirect.com/science/article/pii/S030439912400041X},
author = {Joel Yeo and Benedikt J. Daurer and Dari Kimanius and Deepan Balakrishnan and Tristan Bepler and Yong Zi Tan and N. Duane Loh},
keywords = {Diffraction tomography, Phase retrieval, Cryogenic electron microscopy, Single particle reconstruction, Ewald sphere curvature correction},
}
```

## License
Our code is licensed under GNU GPLv3. By downloading the software, you agree to the terms of this License.
