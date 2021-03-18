# Set Function For Time Series
## Paper
* Set Function For Time Series (Horn et al, 2019) [LINK](https://arxiv.org/pdf/1909.12064.pdf)
* Author's code is available

## Codes
    1. SeFT model
    2. Transformer model (it decodes using fully-connected units)
    3. Metrics for evaluation : accuracy, AUPRC, AUROC
    4. Codes for data processing included

# Settings
## Hyperparameters
* Refer paper's *Appendix* please
* SeFT was trained for 60(30+30)epochs

# Experiment Results
## My Experiment Results
|Dataset|Model|Accuracy|AUPRC|AUROC|
|---|---|---|---|---|
|Pysionet2012|Transformer|None|None|None|
|Pysionet2012|**SeFT**|**85.2±1.8**|None|**57.3±6.9**|

## Paper's Experiment Results
![results](https://user-images.githubusercontent.com/64223259/110726436-b928e800-825c-11eb-9982-4c867d05385c.png)

# Copyright
## for dataset
* MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available from: http://www.nature.com/articles/sdata201635
* Open Data Commons Attribution License (ODC-By) v1.0 : https://www.physionet.org/content/challenge-2012/1.0.0/