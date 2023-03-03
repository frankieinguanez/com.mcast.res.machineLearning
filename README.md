# com.mcast.res.machineLearning
A machine learning repository for machine learning content.

# Setting up on Windows 11 with native GPU support.
- Download and install latest [Nvidia Video Driver](https://www.nvidia.com/download/index.aspx)

**P.S.** Restart PC after this step

- Download and install [cuda](https://developer.nvidia.com/cuda-downloads)

**P.S.** Restart PC after this step

- Download and install [Visual C++ via Microsoft Visual Studio Community](https://visualstudio.microsoft.com/vs/community/), install C++ developer package
- Download and install [Anaconda](https://anaconda.org/)/[MiniConda](https://docs.conda.io/en/latest/miniconda.html)

- In command prompt check cuda version by running `nvcc -V`
- Go to [PyTorch](https://pytorch.org/get-started/locally/) to get the command for your compatible version. Revise conda environment command below.
- Create conda environment

```
conda create --name ml python=3.10
conda activate ml
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install "tensorflow<2.11"
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge scipy
conda install -c conda-forge tqdm
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
conda install -c conda-forge jupyter
conda clean --all
```

- To verify if PyTorch is using GPU type the following command that utilize [torch package](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) in the Anaconda prompt with the respective conda environment activated:

```
python

import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
```

# Useful links
- [Google Remote Desktop](https://remotedesktop.google.com/)
