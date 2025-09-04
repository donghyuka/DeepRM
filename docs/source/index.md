# DeepRM
Deep learning for RNA Modification

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-red?logo=github)](https://github.com/vadanamu/DeepRM)
[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![GitHub Repo stars](https://img.shields.io/github/stars/vadanamu/DeepRM?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/vadanamu/DeepRM)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/vadanamu/DeepRM)
![GitHub contributors](https://img.shields.io/github/contributors/vadanamu/DeepRM)
![GitHub language count](https://img.shields.io/github/languages/count/vadanamu/DeepRM)

```{toctree}
:maxdepth: 1
installation
quickstart
usage
cli/index
api/index
advanced-installation
troubleshooting
license
```

## ✨ Introduction
DeepRM is a deep learning-based framework for RNA modification detection using Nanopore direct RNA sequencing.
This repository contains the source code for training and running DeepRM.

## 🎯 Key Features
* **High accuracy**: Achieves state-of-the-art accuracy in RNA modification detection and stoichiometry measurement.
* **Single-molecule resolution**: Provides single-molecule level predictions for RNA modifications.
* **End-to-end pipeline**: Easy-to-use, 2-step workflow from raw reads to site-level & molecule-level predictions.
* **Customizable**: Supports training of custom models.


## 📝 Citation
If you use DeepRM in your research, please cite the following paper:
```{code-block} text
:class: nohighlight
@article{
  title={Comprehensive single-molecule resolution discovery of m6A RNA modification sites in the human transcriptome},
  author={Gihyeon Kang, Hyeonseo Hwang, Hyeonseong Jeon, Heejin Choi, Hee Ryung Chang, Nagyeong Yeo, Junehee Park, Narae Son, Eunkyeong Jeon, Jungmin Lim, Jaeung Yun, Wook Choi, Jae-Yoon Jo, Jong-Seo Kim, Sangho Park, Yoon Ki Kim, Daehyun Baek},
  journal={In review},
  year={In review},
  publisher={In review}
}
```

## 📐 Architecture
![deeprm_architecture.png](../images/deeprm_architecture.png)


## 🏛️ Contributors
This repository is developed and maintained by the following organization:
* **Laboratory of Computational Biology, School of Biological Sciences, Seoul National University**
    * Principal Investigator: Prof. Daehyun Baek
* **Genome4me, Inc., Seoul, Republic of Korea**


## 🏛️ Acknowledgements
This study was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT, Republic of Korea (MSIT) (RS-2019-NR037866, RS-2020-NR049252, RS-2020-NR049538, and RS-2022-NR067483), by a grant of Korean ARPA-H Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (RS-2025-25422732), by Artificial Intelligence Industrial Convergence Cluster Development Project funded by MSIT and Gwangju Metropolitan City, by National IT Industry Promotion Agency (NIPA) funded by MSIT, and by Korea Research Environment Open Network (KREONET) managed and operated by Korea Institute of Science and Technology Information (KISTI).
