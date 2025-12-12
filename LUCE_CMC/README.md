# Learning Dual Enhanced Representation for Contrastive Multi-view Clustering



> Abstract: Contrastive multi-view clustering is widely recognized for its effectiveness in mining feature representation across views via contrastive learning (CL), gaining significant attention in recent years. Most existing methods mainly focus on the feature-level or/and cluster-level CL, but there are still two shortcomings. Firstly, feature-level CL is limited by the influence of anomalies and large noise data, resulting in insufficient mining of discriminative feature representation. Secondly, cluster-level CL lacks the guidance of global information and is always restricted by the local diversity information. We in this paper Learn dUal enhanCed rEpresentation for Contrastive Multi-view Clustering (LUCE-CMC) to effectively addresses the above challenges, and it mainly contains two parts, i.e., enhanced feature-level CL (En-FeaCL) and enhanced cluster-level CL (En-CluCL). Specifically, we first adopt a shared encoder to learn shared feature representations between multiple views and then obtain cluster-relevant information that is beneficial to the clustering results. Moreover, we design a reconstitution approach to force the model to concentrate on learning features that are critical to reconstructing the input data, reducing the impact of noisy data and maximizing the sufficient discriminative information of different views in helping the En-FeaCL part. Finally, instead of contrasting the view-specific clustering result like most existing methods do, we in the En-CluCL part make the information at the cluster-level more richer by contrasting the cluster assignment from each view and the cluster assignment obtained from the shared fused features. The end-to-end training methods of the proposed model are mutually reinforcing and beneficial. Extensive experiments conducted on multi-view datasets show that the proposed LUCE-CMC outperforms established baselines to a considerable extent.

**If you found this code helps your work, do not hesitate to cite my paper or star this repo!**

## Installation
Requires Python >= 3.8 (tested on 3.8)

To install the required packages, run:
```
pip install -r requirements.txt
```



## Running an experiment
In the `src` directory, run:
```
python -m models.train -c <config_name>
```
where `<config_name>` is the name of an experiment config from one of the files in `src/config/experiments/`. eg. python -m models.train -c caltech3v

## Evaluating an experiment
Run the evaluation script:
```Bash
python -m models.evaluate -c <config_name> \ # Name of the experiment config
                          -t <tag> \         # The unique 8-character ID assigned to the experiment when calling models.train
                        
```
