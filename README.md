# Adaptive Agent Transformer for Few-shot Segmentation

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

In this paper, authors aim to address a few-shot segmentation (FSS) problem. They offer a new transformer-based architecture, called "Adaptive Agent Transformer" (AAFormer) that claims to surpass many state-of-the-art models, and it was published at ECCV 2022 conference. Our aim is to implement the AAFormer architecture with the guidance of the original paper and its supplementary material, and compare the results for reproducibility.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

The objective of few-shot segmentation (FSS) is to segment objects in agiven query image with the support of few sample images. The major complication of FSS is the utilization of the limited information the support images incorporates. Some of the methods in the literature adopt prototypical learning or affinity learning strategies. Prototypical learning methods use masked average pooling to achieve a single prototype  to anticipate outperforming with noisy pixels while the affinity learning methods attempt to leverage pixel-to-pixel similarity between support and query features for segmentation. The proposed method in the paper (**AAFormer**) integrates the adaptive prototypes as agents into affinity based FSS via a transformer encoder-decoder architecture. The transformers architecture has three main parts. The first part is the **Representation Encoder** which is very similar to the encoder part of the standard [transformer](https://arxiv.org/abs/1706.03762) structure which employs the self-attention mechanism for the query and support features seperately and outputs the encoded support and query features to be fed to the **Agent Learning Decoder**. This is one of the two decoders in the model which injects the support information into learing agents to direct the information gathered with support images to the query image. The other decoder is the **Agent Matching Decoder** which yields the retrieved features after crossing the agents tokens with support and query features and alligning the outputs of them.    

# 2. The method and our interpretation

## 2.1. The original method

@TODO: Explain the original method.

- **Initial Agent Tokens**: 
Adaptive Agent Transformer, takes an initial set of agent tokens at its Agent Learning Decoder stage. These tokens are initialized according to Algorithm 1, provided by the paper's supplementary material. The tokens are initialized by utilizing the n-shot mask information of the support images such that it will provide a good representation to bridge the information gap between query images and support images at the Agent Learning Decoder module. The tokens are initialized by selecting indices from support features. Specifically, the feature location where the foreground and background pixels' distance optimized is selected for every individual token.

- **Agent Learning Decoder**: takes Initial Agent Tokens, Support Masks, as well as Support Features obtained from Representation Encoder as its input. The aim is to produce Agent Tokens that will bridge the gap between query features and support features such that these features can be aligned by Agent Matching Decoder in the next stage. In order to obtain these Agent Tokens, Initial Agent Tokens are fed into a masked cross-attention module together with Support Features in eqn.3. Then, authors use Optimal Transport (OT) algorithm that condenses the set of agent tokens such that the tokens will be optimally different from each other.


## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

Throughout our source code, we have discussed our interpretation and assumptions in detail at the comments starting with "`# Assumption:`".


* **Initial Agent Tokens**: There are several unclear parts of Algorithm 1 of the supplementary material. First, the definitions $X$ and $L$ are not clear. They are claimed to be the foreground and background pixels' locations set, yet there is no specification about how to obtain them. It can be trivially inferred the masks are where the foreground pixels exists; however, Algorithm 1 is supposed to work in feature space, not the original image space. Therefore, we assume the masks should be interpolated to feature space to obtain foreground pixel locations for $X$. Second, the selection of $x$ in line 3 seems to be unclear, in which we have implemented Algorithm 1 twice to see if one of our assumptions will work. The further information about these assumptions can be tracked from our comments at `tokens.py`.

* **Agent Learning Decoder**: The major change we have in this part is the interpretation of eqn.7. When we trackdown the matrices' dimensionalities, eqn.7 should be $FFN(SV^s)$ instead of $FFN(S)V^s$. We support our claim throughout the comments at `agentlearningdecoder.py` in detail. We assumed that there is a typo in eqn.7, since the same description of eqn.7 is also provided for eqn.12 (and there is a typo in eqn.12 description, $V^s$ is supposed to be $V^q$, which is the same with our claim.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

We have implemented the setup of the original paper as closely as we could. The settings we have changed can be reviewed from the source code comments in detail. The paper uses 473 image resolution; however, we set the resolution to a lower value, i.e. 128 for our experiments. We provide our hyperparameters explicitly in `main.ipynb` and their values provided by the original paper and state the hyperparameters that are not mentioned in the paper.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

Our main file is `main.ipynb` where we declare step by step code cells to run our code. Please refer to "Prepare Dataset" section's comments to review the steps of downloading and placing the dataset. PASCAL VOC2012 dataset can be downloaded [from here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and the SBD extension is provided by [here](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view) [2]. Extract the SBD extension's "SegmentationClassAug" folder inside "VOC2012" folder, and put "VOC2012" folder inside a folder named "Datasets". Finally, "Datasets" folder should be in the "AAFormer" folder (which contains this repo's source) if you are working on Colab, or it should be in the same directory with "AAformer" (i.e. not inside of AAFormer) if you will run the notebook on your local. After setting up the dataset, you can run the notebook smoothly.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.



# 5. References

@TODO: Provide your references here.
[1] AAFormer
[2] HSNET
[3] POT
[4] ...

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.

Yusuf Soydan, yusuf.soydan@metu.edu.tr
Bartu Akyürek, bartu.akyurek@metu.edu.tr

