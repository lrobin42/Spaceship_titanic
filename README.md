## Project Premise 

This project was created to document modeling approaches used during the [Spaceship Titanic Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic). 

## Project Approach

A few different approaches were taken during this competition, based off the starting assumption that ensemble models would give us a leg up on predicting labels within a blind,label-less test set. After our initial analysis and data cleaning classifiers were trained using TPOT, LGBM, XGB, CatBoost, and AdaBoost packages, before moving to stacking and voting classifiers. 

In none of these cases did our models perform to the level needed. From there we moved to KNN and Label Spreading classifiers before having more success with the the Tensorflow Random Forest model. This model reached our target accuracy, and is the final model used in this competition. 

## Project Findings
All supervised learning classifiers chosen for our initial set of 5 ensembles performed at 75-78% accuracy on the test set, with test accuracy ranging from 23-68%. A stacking classifier trained on our basket of ensembles outperformed a similarly trained voting classifier, and the Label Spreading classifier was our highest performing model outside of the tensorflow decision forest. 

Though much more computationally intensive, the tensorflow model was our best performing classifier set, coming in at 79.284% accuracy.

## Recommendations for further research
In future iterations of this project, a deeper dive into the use of other tensorflow models or more manually calibrated hyperparameters might give us a little more performance on the accuracy front. Additionally, we could explore how to chain supervised and unsupervised classifiers in order strengthen weaker classifiers similar to how we did with a stacking classifier in this project.

Lastly, a different approach could be to restrict models to only model types that can be feasibly trained without GPU resources, but require the same level of accuracy for passing submissions.

## Relevant Files
Please check out the spaceship_functions.py file to see the helper functions used, and requirements.txt for module/package information. Notebooks are broken into two parts, so please check out the (un)supervised_learning_models.ipynb before tensorflow_modeling.ipynb in order to see the EDA, data wrangling, and modeling progression described above.
