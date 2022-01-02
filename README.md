# Reinforcement Learning for Dialogue Generation

**This is the repo for ELEN 6885 Reinforcement Learning Project.**

We reimplement the [paper](https://arxiv.org/pdf/1606.01541.pdf) _Deep Reinforcement Learning for Dialogue Generation_ in PyTorch. Compared to the original paper, we changed the dataset.

Authors: Lisen Dai, Xiangcong Kong, Yixian Cheng, Rui Sun. [Report](https://drive.google.com/file/d/1LekHzvvnSrQujVMJp07_jRkdoBfxctUm/view?usp=sharing).

## Abstract

> Recently, neural models in dialogue generation have shown great promise for generating responses in conversational agents. But they tend to be shortsighted and predicting utterances one at a time while ignoring their impact on future outcomes. As a result, predicting the future content of a dialogue is significant to generate coherent, rational and interesting dialogues. It shows a need to let traditional NLP dialogue models be combined with reinforcement learning. In this paper, we show how these goals are integrated, which means we apply deep reinforcement learning to model future reward in chatbot dialogue. The model simulates dialogues between two virtual agents, using policy gradient methods to reward sequences that display some important conversational properties: informativity, coherence, and ease of answering. Then, we evaluate our model on diversity, length as well as with human judges, showing that the proposed algorithm generates more interactive responses and manages to foster a more sustained conversation in dialogue simulation.
>

## Installation

* Clone this repo, and we'll call the directory that you cloned as ${DRL4DG_ROOT}
* Install dependencies. We use python 3.8 and pytorch >= 1.7.0

```
conda create -n DRL4DG
conda activate DRL4DG
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd ${DRL4DG_ROOT}
pip install -r requirements.txt
```

## Dataset

High-quality first sentence as the starting of dialogue is important to the simulation. Some sentences, such as "What?", are not good enough for the simulation. Those are so vague and lack of context, making it confusing to answer. To avoid this, we choose a very standard dataset named "**DailyDialog**". **DailyDialog** is a high-quality multi-turn dialog dataset, human-written and less noisy. Not only the dialogue text, it also has labels of communication intention and emotion information. However, in our experiments, we only use the texts.

Apart from this, we also use the scripts from **Rick and Morty** to train our model. We found the dataset on Kaggle. It might be a little bit noisy but more likely to be daily.

## Training

We trained the model on one GeForce GTX 1080 Ti GPU. Training on GPUs is recommended.

```
cd ${DRL4DG_ROOT}
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Experimental Results

![image](https://user-images.githubusercontent.com/36061421/147042392-9c6f03ca-47ff-470d-827a-03087654a408.png)

![image](https://user-images.githubusercontent.com/36061421/147042409-99cbcfb9-b901-4dba-8138-4742dd3753b9.png)

![image](https://user-images.githubusercontent.com/36061421/147042429-12d0295a-3ef3-46c6-801d-0d9d6dc54335.png)

## Acknowledgement

Thanks for the assistance from Prof.Li and Prof.Wang. Also, thanks for every member's commitment.
