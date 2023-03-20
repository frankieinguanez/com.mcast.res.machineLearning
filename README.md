# com.mcast.res.machineLearning
A machine learning repository for machine learning content.

# Setting up
- [GIT](https://git-scm.com/): Git libraries for collaboration.
- [GitHub Desktop (Optional but recommended)](https://desktop.github.com/): a git client with UI if you prefer.
- [Python](https://www.python.org/): either through package manager (Windows Store, Apple App Store, Ubuntu Repository) or directly from site.
- [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html): this is optional if using CPU processing but convenient. Mandatory if using GPU processing.
- [VSCode (Optional but recommended)](https://code.visualstudio.com/): recommended IDE for development.

Post installation of application use the scripts found under the setup folder to create a virtual environment, conda environment (optional) and to install the recommended VSCode extensions. You may want to consider setting up Run Configurations in VSCode accordingly.

# Setting up PyTorch and CUDA on Windows 11 with native GPU support.
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

- Verifying GPU usage

Open a command prompt and type the following code:

`nvidia-smi`

**P.S.** During computation, if using GPU you should see a task using the GPU. It is suggested to have OS reserve GPU for intensive processing.

# Setting up CUDA, TensorFlow/Keras with Native GPU support (NVIDIA-CUDA)
Adapted from Jeff Heaton's guide on [YouTube](https://www.youtube.com/watch?v=OEFKlRSd8Ic)/[GitHub](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup2.ipynb)

- **Step 01** - Install [Nvidia Video Driver](https://www.nvidia.com/download/index.aspx)

**P.S.** Restart PC after this step.

- **Step 02** - Install [Visual C++ Package via Microsoft Visual Studio Community](https://visualstudio.microsoft.com/vs/community/)

- **Step 03** - Install Python 3.10 from [homepage](https://www.python.org/) or via Windows Store
- **Step 04** - Install [Cuda](https://developer.nvidia.com/cuda-downloads)

**P.S.** Restart PC after this step.

- **Step 05** - Install [Anaconda](https://anaconda.org/)/[MiniConda](https://docs.conda.io/en/latest/miniconda.html), setup conda environment, Jupyter, create Jupter Kernel, [Tensorflow/Keras](https://www.tensorflow.org/install/pip)

Launch Anaconda Prompt and **run as administrator**. Run the following code.

```
conda create --name tf python=3.10
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
conda install -c conda-forge numpy
conda install -c conda-forge jupyter
ipython kernel install --name "tf-kernel" --user
conda install -c conda-forge matplotlib
conda install -c conda-forge pandas
conda install -c conda-forge scikit-learn
conda clean --all
conda env export > tf.yaml
conda deactivate
```

- **Step 06** - Test if tensorflow version is correct and if Cuda is using GPU

Launch Anaconda Prompt and **run as administrator**. Run the following code.<br/>

```
conda activate tf
python -c "import tensorflow as tf; print(tf.__version__)"
```

**P.S.** Should show 2.10.1.<br />

`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`<br />

**P.S.** Should give you a list of supported graphic cards.

- **Step 07** - Testing tensorflow and verifying GPU usage

Open a command prompt and type the following code:

`nvidia-smi`

This should show a table with the list of supported graphic cards. Note the device number given such as GPU 0 and the GPU-Util value. Run the `tensorflow_get_started.ipynb` notebook using the tf conda environment. This notebook processes the mnist dataset. During model fitting run the `nvidia-smi` command again and note the GPU-Util. This is a more accuracte representation than the task manager performance visual. For more information check the TensorFlow [Get Started](https://www.tensorflow.org/tensorboard/get_started) page.

# Useful links
- [Anaconda repository](https://anaconda.org/anaconda): to determine correct command for installation of package via conda repository.
- [scikit-learn](https://scikit-learn.org/stable/install.html): for further research on machine learning related modules.
- [Cuda, tensorflow, keras setup - video](https://www.youtube.com/watch?v=OEFKlRSd8Ic): Non Anaconda guide, low level installation of Cuda, tensorflow, keras for machine learning with GPU 
- [Cuda, tensorflow, keras setup - GIT](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup2.ipynb)
- [Python Deep Learning Course](https://github.com/jeffheaton/t81_558_deep_learning)
- [Google Remote Desktop](https://remotedesktop.google.com/)

# Setting remote desktop on Debian/Ubuntu systems
-Install NoMachine on both devices. This will enable connection over a local network out of the box-For over the internet connections, open 4000 port on tcp/udp on the desktop top be connected to. Then use the public IP of the remote desktop and 4000 as port for an easy connection.
-For portforwarding guides: [How To Set Up Port Forwarding - Port Forward](https://portforward.com/)
