# Kaggle Challenge

Predicting blood glucose fluctuations is a vital aspect of managing type 1 diabetes, as it helps improve treatment outcomes and daily quality of life. This project, inspired by a Kaggle Challenge, focuses on developing advanced algorithms to better anticipate glucose level changes, addressing the challenges faced by individuals living with this condition.

## Goal

Forecast blood glucose levels one hour ahead using the previous six hours of participant data.

## Description

### Type 1 Diabetes

Type 1 diabetes is a chronic condition in which the body no longer produces the hormone insulin, making it unable to regulate blood glucose (sugar) levels. Without careful management, this can become life-threatening, therefore, the patients with this condition must inject insulin to manage their blood glucose levels. Many different factors, including eating, physical activity, stress, illness, sleep, alcohol, and many more, impact blood glucose levels, making insulin dosage calculations complex. The constant need to consider how an action may impact blood glucose levels and how to counteract them places a significant burden for those with type 1 diabetes.

An essential part of managing type 1 diabetes is predicting how blood glucose levels will change over time. While various algorithms have been developed for this purpose, the untidy nature of health data measurements and numerous unmeasured factors limit their effectiveness and accuracy. This competition aims to advance this work by challenging participants to predict future blood glucose using a newly collected dataset.

### The Dataset

The dataset used in this competition is part of a newly collected, real-world data from young adults in the UK whith type 1 diabetes. All participants used continuous glucose monitors and insulin pumps, and a smartwatch was provided to collect activity data during the study. The complete dataset will be published after the competition for research purposes. Additional details about the study can be found in this [blog post](https://jeangoldinginstitute.blogs.bristol.ac.uk/2024/08/19/how-smartwatches-could-help-people-with-type-1-diabetes/).

### Evaluation

Submissions are evaluated on Kaggle based on Root Mean Square Error (RMSE) between the predicted blood glucose levels an hour into the future and the actual values collected at that time.

RMSE is defined as: 

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

where 𝑦̂ 𝑖 is the 𝑖-th predicted value, 𝑦𝑖 is the 𝑖-th true value and 𝑛 is the number of samples.

The RMSE is computed from the <code>bg+1:00</code> (future blood glucose) predictions in the submission file, comparing them to the actual future blood glucose values. For both public and private leaderboards, RMSE values are calculated using unknown, non-overlapping samples from the submission file across all participants.
