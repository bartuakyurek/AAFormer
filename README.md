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

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

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

