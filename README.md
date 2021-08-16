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

Throughout the competition our team tried multiple different methods to arrive at a solution.  The below instructions show how to utilize our code to generate different types of results:

### Baseline GPT2 Code

As provided, the starter code utilizes GPT2 to solely solve the learning math problem. First, the model can be trained by running the following:

```bash
python src/train_gp2_model.py --MATH-dataroot "./dataset/train/*/*" --save-steps 10 --epochs 2 --batch-size-per-replica 4 --arch distilgpt2 --grad-acc-steps 1
```

Next, the trained model can be evaluated to produce the `predictions.csv` file by running (where the --load flag is changed to match the path of the trained model):

```bash
python src/eval_gpt2_model.py --load "./checkpoints/TEMP/08-14-2021__14:31:24/final_checkpoint/" --math-dataroot "./dataset/test/*/*" --arch distilgpt2
```

### Other Transformer Code

In addition to GPT2, we also tested using different transformer only architectures by changing the starting scripts as needed to work with these formats.  The following scripts are provided, which can be trained and evaluated in a similar manner as the GPT2 starter code:

```bash

## BERT and Roberta
python src/train_bert_model.py -h

python src/eval_bert_model.py -h

## T5
python src/train_t5_model.py -h

python src/eval_t5_model.py -h

```

### Graph2Tree Code

In order to train a model, the following command can be used:

```bash
cd src
python trainer.py

```

The command can be run with the -h flag to see the various parameters that can be set such as the location of the dataset, the learning rate, max number of epochs, and beam search parameters.

### Graph2Tree with Transformer Code

Another promising path was to build off of the provided code to combine transformers with a graph2tree methodology.  To do this, first we can run the following to process and tokenize the dataset with the T5 transformer:

```bash
python src/export_dataset_t5.py --tokenizer t5 --mode train
```

After this is run, there is a pickle file generated which contains the processed dataset (`t5-math-data.pickle`).  Once created, a jupyter notebook server can be started, and the user can execute the `src/custom-graph2tree-t5.ipynb` file which will load the dataset and use the provided code to begin training based on a combination of the provided code and some custom functions to fill in the missing pieces:

```bash
jupyter notebook ./src

```




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
