# MSI-IIE

![](pictures/workflow.png)

MSI-IIE is a deep learning-based framework for cross-sample ion image integration of mass
spectrometry imaging (MSI). Mass spectrometry imaging (MSI) generates thousands of ion images
from various metabolites. However, integrating these spatial patterns across multiple
MSI samples remains a major challenge. Here, we developed MSI-IIE, a self-supervised 
deep learning framework that learns cross-sample ion image embeddings. The embeddings
capture spatial features from individual ion images without strict constraints on consistent 
section size and sample acquisition conditions.

## Installation guide
For Windows system, we suggest to use Anaconda to run following code.

1. Clone the source code from GitHub.
```bash
git clone https://github.com/JunYang2021/MSI_IIE.git
cd MSI_IIE
```

2. Create an Anaconda environment with required packages.
```bash
conda env create -f environment.yml
conda activate msi_iie_env
```

## Demo
We provide demo in Jupyter notebook to demonstrate the functions in MSI-IIE. Users can use their own
files to evaluate.
```bash
cd demo_notebooks
jupyter notebook
```