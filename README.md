# latent_artificial_syntax_pub (NOW Construction)

Code for the paper:

*On the relative impact of categorical and semantic information on the induction of self-embedding structures* (Hyperlink is coming soon)

Antoine Venant, Yutaka Suzuki

BriGap-2 (an IWCS 2025 workshop)

## Dependencies

The code was tested in `python 3.12.3` and `pytorch 2.4.0+cu121`.

## Models

We evaluated two models, **URNNG**, [*Unsupervised Recurrent Neural Network Grammars (Kim et al., 2019)*](https://arxiv.org/abs/1904.03746) and **CPCFG**, [*Compound Probabilistic Context-Free Grammars for Grammar Induction (Kim et al., 2019)*](https://arxiv.org/abs/1906.10225), using automatically generated French sentences.

To investigate the effect of corpus size, training was controlled by the number of steps rather than epochs. Accordingly, we modified parts of the training code, but for the most part we relied on the original implementation provided by Kim et al. 

Their code is available at the following GitHub repository: [URNNG](https://github.com/harvardnlp/urnng) and [CPCFG](https://github.com/harvardnlp/compound-pcfg).

## Data

We applied a PCFG to generate a large number of French sentences with binary tree structures. By adjusting the probabilities of subject and object relative pronouns, we were able to generate sentences with complex multiple center-embedding structures, as well as sentences without such embeddings. To ensure realistic word dependencies, we used [*CamemBERT*](https://huggingface.co/docs/transformers/model_doc/camembert) to estimate the probabilities of subjects and objects given a verb (*selectional restriction*). The generated datasets fall into four categories:

1. Probability constraints on word choice, with center-embedding structures. (`+sr+mce`)[^1]

2. Probability constraints on word choice, **without** center-embedding structures. (`+sr-mce`)

3. **No** probability constraints on word choice, with center-embedding structures. (`-sr+mce`)

4. **No** probability constraints on word choice, **without** center-embedding structures. (`-sr-mce`)

[^1]:(`sr` : *selectional restrictions*, `mce` : *multiple center-embeddings*)

For each category, we prepared training datasets of sizes **3k, 12k, 100k, and 400k**. The validation set consists of 3k sentences, identical across configurations. For fairness, the test set is fixed to the second configuration (`+sr-mce`).

Due to storage limitations, only a subset of the datasets is made available here. Please contact us if you require access to the full datasets.

## Evaluation

## How to use


