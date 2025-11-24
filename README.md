# Split Conformal 
Classification with Unsupervised Calibration

[![Made with!](https://img.shields.io/badge/Made%20with-MATLAB-red)](/AMRC_Matlab)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](#support-and-author)

The provided files implement the proposed method for split conformal prediction with unsupervised calibration samples.


## Source code

[![Made with!](https://img.shields.io/badge/Made%20with-MATLAB-red)](CL-MRC_Matlab)

(/code) folder contains the Matlab files required to execute the method:

*    main.m script that runs the methods presented with the same settings as those in the experimental results shown in the paper using the dataset `USPS' that can be found in the folder '/data'. In addition, the function also obtains results with the conventional approach with supervised calibration samples and the naive approach with unsupervised calibration samples
*    find_quant.m function that finds the conformal quantile using the methods presented
*    select_sigma.m function that selects the bandwidth parameter for the Gaussian kernel used
*    find_p.m function that obtains label probabilities by solving a quadratic optimization problem (using cvx and Mosek solver if variable mosek=1 or using Matlab function if mosek=0)
*    weighted_quantile.m function that determines quantiles for values with corresponding probabilities
*    compute_score.m function that computes values for the adaptive score

## Test case

File main.m obtains set-prediction rules and compute the corresponding coverage probabilities and set sizes for one random partition of USPS dataset.


## Support and Author

Santiago Mazuelas

smazuelas@bcamath.org



## License 

This library carries a MIT license.

## Citation

If you find useful the code in your research, please include explicit mention of our work in your publication with the following corresponding entry in your bibliography:

@inproceedings{Maz:25,
  title      ={Split Conformal Classification with Unsupervised Calibration},
  author     ={Mazuelas, Santiago},
  booktitle  ={{A}dvances in {N}eural {I}nformation {P}rocessing {S}ystems},
  volume     ={38},
  pages      ={},
  year       ={2025},
  month      ={Dec.}
}
