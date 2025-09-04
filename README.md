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

Due to storage limitations, only a subset of the datasets is made available here. 
This repository contains 3k size training dataset, their 3k validation dataset and 75k test dataset. 
(`lex_relobj` for `+sr+mce`, `lex_norelobj` for `+sr-mce`, `flat_relobj` for `-sr+mce` and `flat_norelobj` for `-sr-mce`)

Please contact us if you require access to the full datasets.

## Evaluation

Evaluation was conducted using the F1 score and recall of preterminals.
The F1 score was computed with code based on [PYEVALB](https://github.com/flyaway1217/PYEVALB).

## How to use

The following are sample commands for running the experiments:

1. Pretraitement

The preprocessing is common to URNNG and CPCFG. Executing it once will transform the data into the proper format for both models.

`preprocess.py --trainfile ./data/train_3k_lex_relobj.txt --valfile ./data/val_lex_relobj.txt --testfile ./data/hd-test_lex_norelobj.txt --outputfile ./lex_relobj_3k --vocabsize 10000 --lowercase 1 --replace_num 1 `

2. Training

**URNNG**

`train_.py --train_file ./data/lex_relobj_3k-train.pkl --val_file ./data/lex_relobj_3k-val.pkl --save_path urnng_lex_relobj_3k_0.pt --mode unsupervised --gpu 0 --seed 3435 --print_every 100 --val_every 500 --kl_warmup 10000 --train_q_steps 10000 --min_steps 1500 --num_epochs 250 --early_stop_by_step True  --train_data_output --save_each_val `

**CPCFG**

`train_.py --train_file ./data/lex_relobj_3k-train.pkl --val_file ./data/lex_relobj_3k-val.pkl --save_path cpcfg_lex_relobj_3k_0.pt  --gpu 0 --seed 3435 --print_every 100 --val_every 500 --min_steps 1500 --num_epochs 250 --incr_step 500 --max_length 30 --final_max_length 999 --early_stopping_patience 3 --early_stopping --min_steps 5000  --train_data_output --save_each_val`

For *NPCFG*, you can simply add `--z_dim 0`.

4. Parse

**URNNG**

`parse.py --model_file ./urnng_lex_relobj_3k_0.pt --data ./data/hd-test_lex_norelobj.txt --out_file ./pred-parse_lex_relobj_3k_0.txt --gold_out_file ./gold-parse_lex_relobj_3k_0.txt --gpu 0`

**C(N)PCFG**

`eval.py --model_file ./cpcfg_lex_relobj_3k_0.pt --data_file ./data/hd-test_lex_norelobj.txt --out_file ./pred-parse_lex_relobj_3k_0.txt --gold_out_file ./gold-parse_lex_relobj_3k_0.txt --gpu 0`




