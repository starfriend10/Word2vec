# Word2vec modeling
Share the core code for Word2vec modeling of RCR research papers. The code template only includes the main part and framework for the final Word2vec modeling, while there should include necessary steps of textual data preprocessing before adopting the code. If you use any of the part or believe it is useful, please give credit to:

Zhu, J.-J., & Ren, Z. J. (2023). The evolution of research in Resources, Conservation & Recycling revealed by Word2vec-enhanced data mining. Resources, Conservation & Recycling, 190, 106876. https://doi.org/10.1016/j.resconrec.2023.106876.


Data: The modeling was based on title, abstract, and keywords, but may be modified to include full-text data. Keywords were reamin their original forms (only lowercased), while other texts were tokenized up to quadgram. 

Hyperparameter: min_count and threshold are set, but can be changed or included in the hyperparameter optimization. The provided code was used to search the best model structure of n_feature, window, epochs, and alpha. An example of HPO in the study is shown below:

<img src="https://github.com/starfriend10/Word2vec/blob/main/HPO_scores_m2.jpg" width="1200">

Supervised learning: the code includes two different datasets to supervise the modeling performance. A dataset of similar terms and a dataset of opponenet terms. The model performance is calculated based on the similarty difference.

All the details and explnanation can be found in the published paper. Also check [supporting information](https://github.com/starfriend10/Word2vec/blob/main/Zhu%20and%20Ren%20(2023)%20Supporting%20Information.docx) for additional materials.
