# MSImage2Vec

![](pictures/workflow.png)

MSImage2Vec is a deep learning-based framework for cross-sample ion image integration of mass
spectrometry imaging (MSI). Mass spectrometry imaging (MSI) generates thousands of ion images
from various metabolites. However, integrating these spatial patterns across multiple
MSI samples remains a major challenge. Here, we developed MSImage2Vec, a self-supervised 
deep learning framework that learns cross-sample ion image embeddings. The embeddings
capture spatial features from individual ion images without strict constraints on consistent 
section size and sample acquisition conditions.

## Installation guide
For Windows system, we suggest to use Anaconda to run following code.

1. Clone the source code from GitHub.
```bash
git clone https://github.com/JunYang2021/MSImage2Vec.git
cd MSImage2Vec
```

2. Create an Anaconda environment with required packages.
```bash
conda env create -f environment.yml
conda activate MSImage2Vec_env
```

## Demo
We provide demo in Jupyter notebook to demonstrate the functions in MSImage2Vec. Users can use their own
files to evaluate.
```bash
cd demo_notebooks
jupyter notebook
```