---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Project Home
nav_order: 0
---

<details open markdown="block">
  <summary>
    Home page contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>


# Overview

Sycophancy is the tendency to agree with a statement even when you don't think that it's true.  As people in a social society, we don't always choose to voice our disagreements. We decide whether or not to be agreeable based on factors like our audience, comfort-level, and readiness to engage. We consider the context and decide whether to agree or engage, and we expect others to do the same.

We would not expect Wikipedia to exhibit sycophancy. When we turn to online sources or virtual assistants with a question, we reasonably expect a response that is accurate and consistent with the response that a friend would receive. Today's Large Language Models do not meet this expectation. 

In this post I am digging into the nature of LLM sycophancy. I'm trying to understand scenarios in which it arises - which datasets, output formats, and prompt manipulations most easily encourage the model to output an incorrect or inconsistent response. Next, I dive into two recent approaches that aim to mitigate LLM sycophancy, both of which depend on automated prompt manipulation. Finally, I lay out how either technique would be used in production.  


## Contributions 
- Explore and quantify sycophancy in one particular model and [provide the flexible code]({{site.baseurl}}/exploration-nb) for you to explore new data, models or prompt manipulations.
  - The [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) exhibits significant sycophancy.
  - The behavior sensitive to the specifics of prompt manipulation, the output type (multiple choice vs. freeform), and the dataset.
- Compare two recent methods to address sycophantic behavior in LLMs : 
[Simple synthetic data reduces sycophancy in large language models](https://arxiv.org/abs/2308.03958) involves generating randomly (in)correct opinionated prompts for model finetuning.
 [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827), aka Contrast Consistent Search (CCS), selects a mid-model layer and trains a probe to distinguish truthful from untruthful latent representations.
  - **Similarities**
    - Both methods leverage automated prompt manipulation to generate training data.
    - In generating prompt manipulations, both methods make assumptions about the type of opinion that will be expressed.
    - Both papers were limited to multiple choice questions, but the synthetic data approach may be extendable to summary responses.
  - **Differences**
    - The CCS probe is light-weight and easy to train or update if unexpected opinions arise, where finetuning requires a large GPU and long train-time and would be hard to adapt.
    - CCS does not change the original model and doesn't risk knowledge loss. The CCS probe acts on unchanged model representations, where synthetic-data-finetuning will change results for all queries.
    - CCS on an encoder-decoder model only leverages the encoder and doesn't allow the query to fully pass through the model.
- Suggest methods to leverage these approaches in real-life scenarios.

## Navigating this site


Below you'll find the main project summary. The links and sidepanel will bring you to detailed discussion and code if you'd like to follow along.


# Understanding and Mitigating Model Sycophancy 

## Approaches to address sycophancy in LLMs

In this post I'm digging into two recent publications with methods that might help reduce large language model sycophancy. The first is [Simple synthetic data reduces sycophancy in large language models](https://arxiv.org/abs/2308.03958), and the second is [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).  Both papers link to GitHub repos that were used heavily in this study. 

I chose to work with the [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) model because it is similar to the models used in the papers above, but [the code in this repo](https://github.com/cthomas-ml/syc-syn-latents/tree/main/code) will work with any huggingface model. 


## Quantifying model sycophancy
<span style="color:grey"> 
*Detailed discussion and code for this summary can be found in the Exploring Sycophancy Section in the sidebar.* </span>

I generated two synthetic language datasets using the [synthetic data github repo](https://github.com/google/sycophancy-intervention). I also used an open-source dataset of product reviews.  For all datasets, I processed the prompts to include or exclude incorrect user opinions and limit the model response to multiple choice (Agree or Disagree).  

The model and data and exploration code is available in a very accessible notebook, [try it out!]({{site.baseurl}}/exploration-nb)

In the multiple-choice setting, I tested model accuracy with and without incorrect opinions included in the prompts. The model generally prefers to agree with an opinion expressed in the prompt, even if it answers correctly when no opinion is included. However, it is much easier to elicit sycophancy in the synthetic math dataset than in the IMDB dataset. 


| Dataset | No opinion | Incorrect opinion included  |
---------------|------------|-----------|
| Synthetic Math  |    100%    |     0%    |
| Synthetic Linguistics  |     62%    |     11%   |
| IMDB         |     96%    |     60%   |


When prompted by the synthetic linguistics data with opinions removed, the model performance was only slightly better than random. The aim of finetuning on this dataset is to train the model to prioritize truthfulness over sycophancy. Here I define 'truthfulness' as reflective of the model's learned knowledge. 

To ensure the model will prioritize its knowledge over agreement, the dataset is filtered to include only the 62% of samples that the model answers correctly when opinions are removed. These examples are identified as ones where the model knows the correct answer and can be trained to ignore opinions.

But the synthetic linguistics prompts are long and convoluted and the questions are 2-choice, so getting the right answer doesn't mean the model knows the right answer. In many cases that pass through the filter, the model doesn't have learned knowledge to prioritize. Instead, it is finetuned on meaningless phrases that contain opinions. **Finetuning on these data may reduce sycophancy at the cost of model performance on unrelated, meaningful tasks.** 


I also found that the **sycophantic tendency of the model is sensitive to how an opinion is phrased, the requested output task (classification vs. summarization), and the specifics of the dataset.** For example, in the math dataset the model exhibits complete sycophancy in the multiple choice setting, but not in summary mode, whereas in the IMDB dataset, it is less prone to exhibit sycophancy in the multiple choice setting and can be tricked more readily in summary mode. 

**The sensitivity of the model's sycophancy to the specifics of prompt manipulation will impact both mitigation methods.** Both methods involve prompt editing to create training data and may thus be vulnerable if manipulations don't capture the target sycophancy-inducing prompts. In CCS, the aim is to generate model embeddings for *both* answers to a given question. A probe is trained to contrast those two embeddings and determine which one is the 'truth.' In finetuning with synthetic data, the dataset is generated with a predefined format. 

## Training sycophancy mitigators
<span style="color:grey"> 
*This section benefited greatly from the well-documented and easy-to-follow [CCS GitHub repo](https://github.com/collin-burns/discovering_latent_knowledge).
Note that all experiments in this section were performed with low number of samples, and to get reliable results would require re-running at higher N.  Stay tuned* </span>.

In CCS every prompt gets manipulated into two prompts - one representing each answer of the two-choice question. The two samples are then passed through the model to calculate an embedding vector. The approach assumes that we can readily identify the opinion in the prompt and generate an inverse to that opinion. 

From that point I used the vectors and their labels to train a supervised logistic regression model to determine whether the vectors are distinguishable. If they weren't, the unsupervised CCS method wouldn't have a chance. I then removed the labels to train the CCS probe and report the following response accuracies. 



| Dataset       |     Logistic Regression on Latents| CCS  result |
--------------------------------|----------------|----------------|
| Simple Math (synth)           |      100%      |     100%       |
| Linguistics (synth, filtered) |       58%      |      60%       |
| IMDB                          |       92%      |      88%       |

CCS should ideally restore performance to the no-opinion case. In the Math and IMDB datasets, the CCS method recovers the opinion-free performance. In the synthetic linguistics dataset, even logistic regression failed to distinguish positive from negative embeddings.  The poor separation in latent space supports the suspicion that the model is being finetuned on prompts that don't have meaning to the model.

Next I examined how well the approach generalized to other datasets. A mitigation method would be most useful if I could train a model using one dataset, and apply it on new, unseen datasets. 


| Method     |    Train set      |     Test set       |Response Accuracy|
-------------|-------------------|--------------------|----------------|
| CCS |  50% Linguistics  |       50% Linguistics     |   64%    |
| CCS |    Linguistics    |           Math (synth)    |   100%   |
| CCS |    Linguistics    |               IMDB+       |   82%    |

When training on the linguistics dataset, some but not all accuracy is recovered when testing on the opinionated-IMDB (ie. IMDB+) dataset. 


I was not able to get access to a large enough GPU to execute on finetuning using the filtered sycophantic dataset for a direct comparison. I also plan to test both models against Anthropic's synthetically generated eval datasets. *Opportunity.*

## Protecting against model sycophancy

To effectively leverage either the CCS or finetuning on synthetic data method, one would prepare a trained probe or finetuned model using a dataset selected to protect against the sycophantic behavior of interest.  

A deployed model might behave as follows. 

With CCS, given a trained probe, 
```
1. Receive a human-generated prompt with an opinion stated.
2. Generate the affirmative and negative prompts on the fly.
3. Calculate embeddings for both prompts and use the probe to select correct embedding.
4. Find the response. In the multiple choice case, this is simply the label for the embedding. For a summary response, pass the correct embedding through the rest of the model to generate.
5. Return response 
```

With a model finetuned on synthetic data,
```
1. Receive a human-generated prompt with an opinion stated.
2. Return response.
```

The finetuned model has a simpler and less-flexible inference time application. 

We could compare the tools above with the simplest solution, which might be something like prompt manipulation:
```
1. Receive a human-generated prompt with an opinion stated.
2. Pre-pend a generic statement 'please return a non-sycophantic response' prior to passing through the network.
3. Return response.
```

# What's next
- Finetune the model using the filtered sycophantic dataset and make a direct comparison with CCS on larger training sets. 
- Test both approaches against Anthropic's synthetically generated eval datasets. 
- Examine knowledge loss in both approaches.
- Share more of the code used in this project.
- Where in the model is sycophancy happening? In both the math and IMDB datasets, tendency towards sycophancy is different in summary and multiple choice scenarios.


# About the project
This project was contributed by C Thomas as part of the 2024 BlueDot Alignment course. 
I'd like to thank my cohort for their feedback and solidarity, our leader Cara Selvarajah and the entire BlueDot team for running a great program, and a special thanks to Aaron Scher for thoughtful insights and resources. I'm also grateful to the authors of both papers, who made very accessible code and Hugging Face, which seems to make everything possible. 