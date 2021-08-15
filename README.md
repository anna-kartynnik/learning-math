# Learning to Learn Math Competition (COMS 4995 Summer 2021)
## Submission for Hanna Kartynnik (hk3129) and Jeffrey Bennett (jjb2238)

## Overview

The learning to learn math competition provided a real test case for us to apply some of the tools and techniques learned during the Summer 2021 Deep Learning course. In addition to submitting results with our classmates to get on the Kaggle leader board, our team had the following goals:

* Gain practical experience setting up and configuring various machines and environments with GPU resources
* Work on different data processing techniques
* Experiment with different transformer architectures
* Use graph neural networks and the DGL library
* Develop code to split models across multiple GPUs
* Investigate the effectiveness of pre-training on similar datasets



## Hardware Setup 

* Throughout the competition we utilized and configured many different local and cloud resources. The largest models that we tested were used on a local desktop we had access to with 3 NVIDIA Quadro RTX 6000 GPUs. 


## Installation 

To install the python dependencies, first run the following command:

```bash
pip install -r requirements.txt

```

We used CUDA 11 for all of our GPU support. Because of this, it is necessary to build DGL from source in order to work with this version of CUDA. This can be done as follows:

```bash
## Install the necessary linux libraries
sudo apt-get update
sudo apt-get install -y build-essential python3-dev make cmake

## Clone the DGL source code
git clone --recurse-submodules https://github.com/dmlc/dgl.git

## Build with CUDA
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j4

## Install the Python bindings
cd ../python
python setup.py install

## Depending on the system, if the above didn't work we have had success also using the following
pip install install --upgrade dgl-cu111

```

## Running

In order to train a model, the following command can be used:

```bash
cd src
python trainer.py

```

The command can be run with the -h flag to see the various parameters that can be set such as the location of the dataset, the learning rate, max number of epochs, and beam search parameters.


## Results

TBD


## Resources

The following code bases and repositories were used to complete this competition:

* https://github.com/hendrycks/math
* https://github.com/qinzhuowu/NumS2T
* https://github.com/QinJinghui/SAU-Solver
* https://github.com/2003pro/Graph2Tree
* https://www.kaggle.com/c/learning-to-learn-math/discussion/264692

Additionally, we also looked at incorporating the following datasets to use for pre-training:

* [AMPS](https://drive.google.com/file/d/1hQsua3TkpEmcJD_UWQx8dmNdEZPyxw23/view?usp=sharing)


## Papers

```bibtex
@inproceedings{xie2019goal,
  title={A Goal-Driven Tree-Structured Neural Model for Math Word Problems.},
  author={Xie, Zhipeng and Sun, Shichao},
  booktitle={IJCAI},
  pages={5299--5305},
  year={2019}
}
@article{lample2019deep,
  title={Deep learning for symbolic mathematics},
  author={Lample, Guillaume and Charton, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:1912.01412},
  year={2019}
}
@inproceedings{zhang2020graph,
  title={Graph-to-tree learning for solving math word problems},
  author={Zhang, Jipeng and Wang, Lei and Lee, Roy Ka-Wei and Bin, Yi and Wang, Yan and Shao, Jie and Lim, Ee-Peng},
  year={2020},
  organization={Association for Computational Linguistics}
}
@article{li2020graph,
  title={Graph-to-Tree Neural Networks for Learning Structured Input-Output Translation with Applications to Semantic Parsing and Math Word Problem},
  author={Li, Shucheng and Wu, Lingfei and Feng, Shiwei and Xu, Fangli and Xu, Fengyuan and Zhong, Sheng},
  journal={arXiv preprint arXiv:2004.13781},
  year={2020}
}
@article{pikekos2021measuring,
  title={Measuring and Improving BERT's Mathematical Abilities by Predicting the Order of Reasoning},
  author={Pi{\k{e}}kos, Piotr and Michalewski, Henryk and Malinowski, Mateusz},
  journal={arXiv preprint arXiv:2106.03921},
  year={2021}
}
@article{hendrycks2021measuring,
  title={Measuring mathematical problem solving with the math dataset},
  author={Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2103.03874},
  year={2021}
}
@article{liang2021mwp,
  title={MWP-BERT: A Strong Baseline for Math Word Problems},
  author={Liang, Zhenwen and Zhang, Jipeng and Shao, Jie and Zhang, Xiangliang},
  journal={arXiv preprint arXiv:2107.13435},
  year={2021}
}
@article{griffith2021solving,
  title={Solving Arithmetic Word Problems with Transformers and Preprocessing of Problem Text},
  author={Griffith, Kaden and Kalita, Jugal},
  journal={arXiv preprint arXiv:2106.00893},
  year={2021}
}
@article{meng2019solving,
  title={Solving math word problems with double-decoder transformer},
  author={Meng, Yuanliang and Rumshisky, Anna},
  journal={arXiv preprint arXiv:1908.10924},
  year={2019}
}
```
