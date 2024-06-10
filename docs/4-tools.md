---
layout: page
title: Applications
permalink: /do-something/
nav_order: 4
---


# Protecting against sycophants

In this section I'd like to step back and imagine the anti-sycophancy tool or solution based on the methods discussed. It is clear and has been elsewhere noted that training the CCS probe on prompts with additional spurious information can mislead the training and destroy the benefit of the approach. 

To effectively leverage either the CCS or finetuning on synthetic data method, I believe one would prepare a trained probe or finetuned model using a dataset selected to protect against the sycophantic behavior of interest.  

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

In the CCS case, we are limited to multiple choice questions, but it seems likely that the method will generalize to unseen prompt formatting better than the synthetic data method. 
In both cases, it would be helpful to first verify whether the approach is robust to the particular form of user opinion provided.  

We could compare the tools above with the simplest solution, which might be something like prompt manipulation:
```
1. Receive a human-generated prompt with an opinion stated.
2. Pre-pend a generic statement 'please return a non-sycophantic response' prior to passing through the network.
3. Return response.
```