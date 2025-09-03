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

## Evaluation

## How to use


