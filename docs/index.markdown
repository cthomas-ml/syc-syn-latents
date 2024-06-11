---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Project Home
nav_order: 0
---

# Overview

Sycophancy is the tendency to agree with a statement even when you don't believe that it's true.  As people in a social society, we don't always choose to voice our disagreements. We decide whether or not to be sycophantic based on factors including impact, audience, comfort-level, and readiness to engage. 
If we reject the comfort of sycophancy, we might grow from the experience. It works because people don't expect anyone to know everything or to always share what they know. 

We would not, however, expect Wikipedia to exhibit sycophancy. When we turn to online sources or virtual assistants with a question, we reasonably expect a response that is accurate and consistent with the response that a friend would receive. Today's Large Language Models do not meet this expectation. 

In this post I am digging into the nature of LLM sycophancy. I'm trying to understand scenarios in which it arises - which datasets, output formats, and prompt manipulations most easily encourage the model to output an incorrect or inconsistent response. Next, I dive into two recent approaches that aim to mitigate LLM sycophancy, both of which depend on automated prompt manipulation. Finally, I lay out how either technique would be used in production.  


<!-- <details open markdown="block">
  <summary>
    Home page contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details> -->


## Contributions 
- Explore and quantify sycophancy in one particular model and [provide the flexible code]({{site.baseurl}}/exploration-nb) for you to explore new data, models or prompt manipulations.
  - The [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) exhibits significant sycophancy.
  - Sycophancy is sensitive to the specifics of prompt manipulation, the output type (multiple choice vs. freeform), and the dataset.
- Compare two recent methods to address sycophantic behavior in LLMs 
[Simple synthetic data reduces sycophancy in large language models](https://arxiv.org/abs/2308.03958), and 
 [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827), aka Contrast Consistent Search (CCS), which selects a mid-model layer and trains a probe to distinguish truthful from untruthful latent representations.
  - **Similarities**
    - Both papers were limited to multiple choice questions, but the synthetic data approach may be extendable to summary responses.
    - Both methods leverage automated prompt manipulation to generate training data.
    - In generating prompt manipulations, both methods make some assumption about the type of opinion that will be expressed.
  - **Differences**
    - The CCS probe is very light weight and easy to train, where finetuning the model requires a large GPU (which I could not access during this project).
    - CCS does not change the main model and does not risk knowledge loss. The CCS probe acts on unchanged model representations, where synthetic-data-finetuning changes results for all queries.
    - Because it is a lightweight probe, CCS may be easier to adapt to new types of sycophancy. In the finetuning approach, the synthetic data are generated with a specific format of opinion and query and used in training. This  effectively bakes-in the sycophancy protection for that format of misleading query, but wouldn't be easily adapted to another. 
- Suggest methods to leverage these approaches in real-life scenarios.

## Navigating this site
{: .no_toc }

Below you'll find a brief summary of the work I've done in this project.  The links and sidepanel will bring you to the detailed analysis and code if you'd like to follow along.


# Summary

## Approaches to address sycophancy in LLMs

In this post I'm digging into two recent publications with methods that might help reduce large language model sycophancy. The first is [Simple synthetic data reduces sycophancy in large language models](https://arxiv.org/abs/2308.03958), and the second is [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).  Both papers link to GitHub repos that were used heavily in this study. 



## Quantifying model sycophancy
<span style="color:grey"> 
*Detailed discussion and code for this summary can be found in the Exploring Sycophancy Section in the sidebar.* </span>

I chose to work with the [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) model because it is similar to the models used in the papers above, but the code provided will work easily with any huggingface model. 
I generated two synthetic language datasets using the [synthetic data github repo](https://github.com/google/sycophancy-intervention). I also used an open-source dataset of product reviews.  For all datasets, I processed the prompts to include or exclude incorrect user opinions and limit the model response to multiple choice (Agree or Disagree).  

The model and data and exploration code is available in a very accessible notebook, [try it out!]({{site.baseurl}}/exploration-nb)


| Dataset | No opinion | Incorrect opinion included  |
---------------|------------|-----------|
| Synthetic Math  |    100%    |     0%    |
| Synthetic Linguistics  |     62%    |     11%   |
| IMDB         |     96%    |     60%   |


In the multiple-choice setting, the model generally prefers to agree with an opinion expressed in the prompt, even if it answers correctly when no opinion is included. 

The tendency of the model to agree is sensitive to how an opinion is phrased, the requested output task (classification vs. summarization), and the specifics of the dataset. For example, in the math dataset the model exhibits sycophancy in the multiple choice setting, but not in summary mode, whereas in the IMDB dataset, it is less prone to exhibit sycophancy in the multiple choice setting and can be tricked more readily in summary mode. 


### Sensitivity to prompt manipulation will impact both methods
{: .no_toc }

Both sycophancy mitigation methods depend on prompt manipulation for training. In CCS, we aim to generate model embeddings for *both* answers to a given question. A probe is trained to contrast those two embeddings and determine which one is the 'truth.' In finetuning with synthetic data, the dataset is generated with a predefined format. 



### The synthetic linguistics prompts are confusing 
{: .no_toc }

The synthetic linguistics dataset has randomized opinions injected into verbose true and false statements. 
The aim in finetuning is to train the model to prioritize truthfulness over sycophancy, where 'truthfulness' means reflective of the model's learned knowledge. 

To ensure the model will prioritize its knowledge over agreement, the dataset is filtered to include only samples that the model answers correctly when opinions are removed. Even without opinions, the model accuracy is only slightly better than random. 
This means that in many cases that pass through the filter, the model doesn't have learned knowledge to prioritize. Instead, it is being finetuned on meaningless phrases that contain opinions. This may reduce sycophancy, but it's likely at the cost of model performance on unrelated, meaningful tasks.


## Training Sycophancy Mitigators
<span style="color:grey"> 
*This section benefited greatly from the well-documented and easy-to-follow [CCS GitHub repo](https://github.com/collin-burns/discovering_latent_knowledge).
Note that all experiments in this section were performed with low number of samples, and to get reliable results would require re-running at higher N.  Stay tuned* </span>.

In CCS every prompt gets manipulated into two prompts - one representing each answer of the two-choice question. The two samples are then passed through the model to calculate an embedding vector. 

From that point we can use the vectors and their labels to train a logistic regression model to determine whether the vectors are easily distinguished. If not, we are unlikely to successfully train a CCS probe. 



| Dataset       |     Logistic Regression on Latents| CCS  result |
--------------------------------|----------------|----------------|
| Simple Math (synth)           |      100%      |     100%       |
| Linguistics (synth, filtered) |       58%      |      60%       |
| IMDB                          |       92%      |      88%       |

In the synthetic linguistics dataset, even logistic regression fails to distinguish positive from negative embeddings.

Next we can examine generaliziation. The method would be useful if we could train a model using some dataset, and use the approach on new, unseen datasets. 


| Method     |    Train set      |     Test set       |Response Accuracy|
-------------|-------------------|--------------------|----------------|
| CCS |  50% Linguistics  |       50% Linguistics     |   64%    |
| CCS |    Linguistics    |           Math (synth)    |   100%   |
| CCS |    Linguistics    |               IMDB+       |   82%    |

When training on the linguistics dataset, some but not all accuracy is recovered when testing on the opinionated IMDB dataset. 


I was not able to get access to a large enough GPU to finetune the model using the filtered sycophantic dataset for a direct comparison. I also plan to test both models against Anthropic's synthetically generated eval datasets. 

## Protecting against model sycophancy

To effectively leverage either the CCS or finetuning on synthetic data method, one would prepare a trained probe or finetuned model using a dataset selected to protect against the sycophantic behavior of interest.  

A deployed model might behave as follows. 

With CCS, given a trained probe, 
```
1. Receive a human-generated prompt with an opinion stated.
2. Generate the affirmative and negative prompts on the fly.
3. Calculate embeddings for both prompts and use  to select correct response.
4. Return response.
```

With a model finetuned on synthetic data,
```
1. Receive a human-generated prompt with an opinion stated.
2. Return response.
```

# About the project
This project was contributed by C Thomas as part of the 2024 BlueDot Alignment course. 
I'd like to thank my cohort for their feedback and solidarity, our leader Cara Selvarajah and the entire BlueDot team for running a great program, and a special thanks to Aaron Scher for thoughtful insights and resources. I'm also grateful to the authors of both papers, who made very accessible code and Hugging Face, which seems to make everything possible. 