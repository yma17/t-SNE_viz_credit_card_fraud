# t-SNE_viz_credit_card_fraud
Project for MLH Local Hack Day: Build (January 10-18, 2021). Script to visualize classification of credit card transactions, represented by 30-dimensional data points, as either fradulent or non-fraudulent in 2 and 3 dimensions using t-distributed stochastic neighbor embedding (t-SNE).

Python libraries required: `numpy`, `matplotlib`, `sklearn`.

To use:
1. Place the file `creditcard.csv` (source: https://www.kaggle.com/mlg-ulb/creditcardfraud) in the root directory of this repository.
2. Run `visualize.py`.

The visualization results can also be found in the files `viz_tsne_2d.png` and `viz_tsne_3d.png`.