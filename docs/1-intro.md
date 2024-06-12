---
layout: page
title: Approaches 
permalink: /intro/
nav_order: 1
---

# Mitigation Methods 

This work compares two approaches to improving large language model accuracy in response to misleading prompts.

[Simple synthetic data reduces sycophancy in large language models](https://arxiv.org/abs/2308.03958), from Google DeepMind in Feb, 2024, shows that as models grow larger, they exhibit increasing tendency to agree with an incorrect prompt despite knowledge of the correct response. The paper addresses this sycophancy with supervised fine-tuning on lots of generated data in which the prompt correctness is random but the ground-truth is accurate, effectively training the model to ignore extraneous information provided in a prompt and provide a known, correct answer. The paper demonstrates that they can improve model accuracy on prompts similar to those used in fine-tuning. The approach requires the engineers to (1) know what types of prompts to protect against, and (2) generate significant volumes of training data to combat each type.

[Discovering Latent Knowledge in Language Models Without Supervision (CCS)](https://arxiv.org/abs/2212.03827) from UC Berkeley at ICLR 2023, focuses instead on extracting the correct response from the model's latent representations. This work also requires prompt manipulation. In this case, an incoming prompt is manipulated in two different ways, yielding one phrase for each possible model response. Each phrase is then passed through part of the model to calculate the latent representation vectors. A lightweight probe is then trained on the latent representations by minimizing the sum of the consistent loss, ensuring that the total probability of responses will sum to 1, and the informative loss, which pushes probabilities towards their extremes of 0 and 1. The approach requires the engineers to (1) know how to manipulate prompts to elicit opposing embeddings, and (2) train the probes for the types of questions that are expected. 
<!-- **Note:** There is an updated repo from Berkeley, rebranding this work as ['Eliciting Latent Knowledge'](https://github.com/EleutherAI/elk). -->



