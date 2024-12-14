# Introduction

This project deals with the prediction of blood glucose levels one hour ahead of time of patients with type 1 diabetes. 
Predicting blood glucose fluctuations is crucial for managing type 1 diabetes. An accurate forecast can help alleviate some of the challenges faced by individuals with the condition.

<!--## Goal

Predict blood glucose levels one hour ahead of time using the previous six hours of participant data.-->

## Description

### Type 1 Diabetes

Type 1 diabetes (T1D) is a chronic disease in which the body is no longer able to produce the hormone insulin. Insulin is required by the body to regulate the amount of glucose (sugar) in the bloodstream. Without treatment T1D results in high blood glucose levels which can cause symptoms like frequent urination, increased thirst, increased hunger, weight loss, blurry vision, tiredness, and slow wound healing. Ultimately high blood glucose levels will be fatal. In order to survive those suffering from T1D need to inject insulin to manage their blood glucose levels. Since also low blood glucose levels are potentially life-threatening and insulin counteracts the level of blood glucose it is important to establish a careful insulin management. There are many other factors that impact blood glucose levels, including eating, physical activity, stress, illness, sleep and alcohol. Hence calculating how much insulin to apply is complex. But the continuous need to think about how an action may impact blood glucose levels and what to do to counteract them is a significant burden for those with T1D.

### The Goal

Therefore developing algorithms which can reliably predict blood glucose levels in the future can play an important role in T1D management. Algorithms of varying levels of complexity have been developed that perform this prediction but the noisy nature of health data and the numerous unmeasured factors that impact the target mean there is a limit to how effective they can be. This project aims to use a newly collected dataset in order to predict blood glucose levels one hour ahead of time of patients with T1D.

### The Dataset

The data used in this project was part of a kaggle competition (https://www.kaggle.com/competitions/brist1d) in which our team also participated. It is part of a bigger newly collected dataset of real-world data collected from young adults in the UK who suffer from T1D. All participants used continuous glucose monitors, insulin pumps and were given a smartwatch as part of the study to collect activity data. It is structured as follows

### Evaluation

Submissions are evaluated on Root Mean Square Error (RMSE) between the predicted blood glucose levels an hour into the future and the actual values that were then collected.

RMSE is defined as: 

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

where $\hat{y}_i$ is the $i$-th predicted value, $y_i$ is the $i$-th true value and $n$ is the number of samples.

The RMSE value is calculated from the bg+1:00(future blood glucose) prediction values in the submission file against the true future blood glucose values. The RMSE values for the public and private leaderboards are calculated from unknown and non-overlapping samples from the submission file across all of the participants.
