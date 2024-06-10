# Comparing sycophancy mitigation techniques

C Thomas -
May, 2024 - BlueDot Alignment

## Introduction

This work compares two approaches to improving large language model accuracy in response to misleading prompts.

[Simple synthetic data reduces sycophancy in large language models](https://arxiv.org/abs/2308.03958), from Google DeepMind in Feb, 2024, shows that as models grow larger, they exhibit increasing tendency to agree with an incorrect prompt despite availability of the correct response. The paper addresses this sycophancy with supervised fine-tuning on lots of generated data in which the prompt correctness is random but the ground-truth is accurate, effectively training the model to ignore extraneous information provided in a prompt and provide a known, correct answer. The paper demonstrates that they can improve model accuracy on prompts similar to those used in fine-tuning, but the resulting model is unlikely to generalize to other types of sycophancy. The approach requires the engineers to (1) know what types of responses to protect against, and (2) generate significant volumes of training data to combat each type.

[Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827) from UC Berkeley at ICLR 2023, focuses instead on extracting the correct response by training an unsupervised probe on the model's latent representations. This work also requires prompt manipulation for training, but here there are two prompts generated for each question - one for each possible response. The probe is trained by minimizing the sum of the consistent loss, ensuring that the total probability of responses will sum to 1, and the informative loss, which pushes probabilities towards their extremes of 0 and 1. The approach requires the engineers to (1) know how to manipulate prompts to elicit opposing embeddings, and (2) train the probes for the types of questions that are expected. 
**Note:** There is an updated repo from Berkeley, rebranding this work as ['Eliciting Latent Knowledge'](https://github.com/EleutherAI/elk).

## Sycophancy protection 
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
