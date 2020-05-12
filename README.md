# CausalInferenceChallenges

This repository contains the code for "Challenges and an Empirical Evaluation Framework forCausal Inference with Natural Language"

# Setup
To follow us exactly:

1. create a conda environment with python 3.6
2. Install the following packages:
  - pytorch v. 0.4.1
  - pytorch_pretrained_bert
  - nltk
  - scikitlearn
3. Make sure reddit post data 'posts.npy' is downloaded and in the data folder

# Instructions for Replication
This code block lists the steps taken to run the same experiments as are included in the paper, and produce plots of the same kind.

For each experiment listed in the AXIS BLOCK in Analyze_results_annotated.ipynb, run the experiment using run_experiment_annotated.py. The settings for a given experiment are given in the filename for a given
axi experiment
e.g.
```
# experiment_1_0_0_0_0_60
python run_experiment.py -exp 1_0_0_0_0 -size 60 -lr <lr> -bs <bs> -n_it <n_it>

# experiment_0_0_0_1_0_60
python run_experiment.py -exp 0_0_0_1_0 -size 60 -lr <lr> -bs <bs> -n_it <n_it>

# experiment_1_0_0_0_0_60_nuser4000
python run_experiment.py -exp 0_0_0_1_0 -size 60 -n_user 4000 -lr <lr> -bs <bs> -n_it <n_it>
```

running commands listed above should return outputs with the correct name format, so running the commands
above (and corresponding ones for all other experiments) should be sufficient to get outputs for analysis.


NOTE: training of HBERT can be quite inconsistent. In practice, it can take multiple starts with 
different learning rates to train an effective model. This also depends heavily on batch size, which
itself depends on the GPU you are running on.
Some good settings to start with:
lr = 0.0001
bs = 512
n_it = 40000

One each experiment runs, go through Analyze_results_annotated.ipynb to run random bootstrap causality experiments on the propensity scores returned by models, and finally produce similar plots to the paper.

# Instructions for adding model

The most simple way to do this will be closely follow the code structure of either logistic regression or HBERT models. For instance, take your model code/training script, and fit it into the structure of the propensity model training function in HBERT_simple.py.

Specifically, it is important for your model to have a 'propensity_model' wrapper like the Hierarchical_BERT_propensity_model class.

It should include a 'fit' function that trains the model, as well as a 'score' function which takes a list of examples in the given data format and returns propensity scores for it.

Finally, if using data of a different format than either raw text (as in HBERT data) or BoW (as in data_chi), you must implement a data script in the format of these 2, and include it in run_experiment_annotated.py
