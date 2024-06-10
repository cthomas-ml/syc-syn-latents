---
layout: page
title: Comparing CCS with finetuning
permalink: /methods/
nav_order: 3
---


# Using CCS on the synthetic sycophant datasets

CCS requires prompt manipulation, so that you can get the embeddings for both the positive and the negative set. In the example in the paper, the prompt manipulation is achieved by prepending a statement about the prompt.  For example, 

```
The following movie review expresses a [negative | positive] sentiment: <imdb_review_instance>
```
In these cases, we consider the scenario where an incorrect user opinion is provided, as in the table above.  That prompt can be manipulated to express the opposite opinion.  In CCS, we then find the latent representation for each these statements and train our consistent and informative probe to learn to distinguish them.

## Results on similar train/test data

Prior to training the probe, we can use supervised logistic regression to check to see whether the representations are sufficiently different. We can then remove labels and run CCS. Splitting each dataset 50/50 and checking supervised logistic regression accuracy of CCS embeddings, and comparing that to unsupervised CCS performance:

| Dataset (50/50 split)      |     Logistic Regression on Latents| CCS  result |
--------------------------------|----------------|----------------|
| Simple Math (synth)           |      100%      |     100%       |
| Linguistics (synth, filtered) |       58%      |      60%       |
| IMDB                          |       92%      |      88%       |

*sample sizes in these results are small

## Discussion
Results in this section are meant to be compared with the scenario where we include incorrect opinions, and should ideally restore performance to the no-opinion case. The approach assumes that we can readily identify the opinion in the prompt and generate an inverse to that opinion.  

In the Math and IMDB datasets, the CCS method recovers the opinion-free performance.  The linguistics dataset has been filtered and only includes the prompts that the model answered correctly when opinions were removed. In theory, the opinion-free performance is 100%, but it seems likely that many of the included samples were correct by chance, and slight modifications in the prompt would have resulted in an incorrect response and removal. The CCS method brings accuracy on opinionated linguistics prompts from 11% to 60%. 

It's also notable that even with only one training step for simple math, the accuracy rises to 100%. This indicates a strong separation in the latent space between true and false math statements. 

## Generalization results: when train and test differ

| Method     |    Train set      |     Test set       |Response Accuracy|
-------------|-------------------|--------------------|----------------|
| CCS |  50% Linguistics  |       50% Linguistics     |   64%    |
| CCS |    Linguistics    |           Math (synth)    |   100%   |
| CCS |    Linguistics    |               IMDB+       |   82%    |
| CCS |     IMDB+ or Linguistics? |  Anthropic evals/sycophancy        |          |

IMDB+ means we have prepended the phrases.
To apply CCS at inference time to the Anthropic set requires some real-time prompt manipulations.  Method - 


<!-- ## Sensitivity to prompt manipulation

When initially exploring sycophancy in the model, we observed that model sycophancy was sensitive to multiple details in the prompt.  Specifically, it was sensitive to how the human opinion was stated, which output format was requested (eg. classification vs. summarization task), and the sensitivity differed based on the type of data (eg. simple math vs IMDB sentiment).

Is the CCS method similarly sensitive to details of prompt manipulation?
The CCS approach requires a positive/negative sample pair for a given prompt. Therefore, any opinionated prompt needs to be manipulated to generate an inverse-opinion sample and any opinion-free prompt needs both opinions added prior to passing through the network. 

In the synthetic linguistics data, the embedding and CCS results were  sensitive to the prompt manipulation.  These prompts contain sentences like, "My name is X, I agree with the statement Y. What is your opinion on the statement Y?"
  - Manipulating the prompt by exchanging the user opinion, "agree <-> disagree," the logistic regression accuracy is ~0.3. The agree/disagree embedding vectors are not readily distinguished in this dataset. 
  - Prepending to the prompt the phrase "The best answer is '(A/B)'", the logistic regression accuracy is 0.58 and CCS is 0.6.
  <!-- - with no filtering and "the best answer is '(A/B)'", the logistic regression is 0.4 -->

<!--
On reflection this makes sense. The CCS approach aims to compare the embeddings for the model response, not for the user opinion. 

I did not test additional manipulations in the synthetic math or IMDB datasets. In synthetic math set, where all user opinions incorrectly agree with a false statement about addition, I generated the pair of prompts by exchanging the user 'agree' with 'disagree.'  In the IMDB dataset, I manipulated all reviews by prepending the statement, 'The following music review is positive/negative.' 

In any case, it is straightforward to identify the prompt manipulations that will yield strong signal.  -->



# Finetuning with synthetic data

The linguistics data were originally generated for finetuning, and was observed to reduce sensitivity to human opinion in the math test set.
- The statements in the linguistics dataset are sufficiently confusing that the model doesn't reliably "know" the answers to the opinion-free prompts. 
- The data are filtered based on whether the model generates the correct answer when the opinion is removed. However, the model accuracy is only slightly better than random and it is likely that the responses are correct by chance, rather than due to model certainty. This is supported by the poor separation in latent space of opposite sentiments on the same dataset. 
 - The data filtering strategy is weak, but the approach still works. 
- When the model is finetuned on the filtered dataset, that is, the examples where it achieved an accurate result (if by chance). 
 -  It may be learning to not be agreeable or to ignore extraneous components of a prompt.
 -  It may not be learning to prioritize 'truthfulness,' though. Here I would define truthfulness as reflective of the model's learned knowledge.  In these finetuning data, for many of the prompts, the model doesn't have learned knowledge to prioritize.  It is only learning not to be agreeable. 


| Method     |    Train set      |     Test set             |Response Accuracy|
-------------|-------------------|--------------------------|----------------|
| Finetuning |  50% Linguistics |  50% Linguistics  |       |
| Finetuning |     Linguistics  |     Math (synth) |       |
| Finetuning |     Linguistics |     IMDB+ |       |
| Finetuning |     Linguistics |     Anthropic |       |

The Linguistics data here are the filtered synthetic set. 

# Comparing the approaches

Pros/cons and discussion of how each generalizes
