# Literature Review

## Introduction

This literature review provides an overview of the existing research on blood glucose prediction.
One of the most import source for this review is a meta study {cite}`Nemat2024` that provides an in-depth
examination of various data-driven models for predicting blood glucose levels (BG) in people with type 1 diabetes.
Accurate BG prediction is critical for improving diabetes management, helping patients avoid dangerous fluctuations
in blood glucose levels, and reducing long-term health complications.

Other studies have also explored different approaches to BG prediction, including machine learning models,
statistical methods, and physiological models. These studies have highlighted the challenges of BG prediction,
such as the high variability of blood glucose levels, the complexity of the underlying physiological processes,
and the need for personalized models that account for individual differences in patients.

## Blood Glucose Prediction Model Approaches

Traditionally, prediction models for BGL have been based on physiological models that rely on extensive knowledge of the body's processes.
However, data-driven approaches, which do not require detailed physiological understanding, have gained prominence.

These model approaches can be classified into three categories:

* time series forecasting (TSF),
* machine learning (ML),
* deep neural networks (DNN)

Despite their promise, comparing the performance of these models has been difficult due to differences in datasets, input variables, and model configurations.

In this meta study released in 2024 in Nature, the authors are comparing the effectiveness of these three approaches using:

* univariate (BG only)
* multivariate inputs (BG, carbohydrate intake, insulin dosages and physical activity)

## Data Sources

The data used in these studies typically come from continuous glucose monitoring (CGM) devices, which provide real-time blood glucose measurements.

A very commonly used data set to compare models and inputs is the OhioT1DM dataset {cite}`Marling2020`, which is publicly available and contains data from 30 patients with type 1
diabetes.

To outline the comparison of the three approaches, the authors of the meta study used the OhioT1DM dataset {cite}`Marling2020` to evaluate the performance of different models in
predicting blood glucose levels.

## Results

### Model comparison

The performance of TSF, ML, and DNN models was compared based on their ability to predict blood glucose levels using univariate and multivariate inputs.

* TSF Model (ARIMA): This model showed stable but relatively low performance compared to the ML and DNN models. The ARIMA model struggled with multivariate input and often
  performed better when using only BGL data. This suggests that classical time series models may not be well-suited for handling complex, multivariate data, especially when
  incorporating exogenous variables like insulin and carbohydrate intake.
* ML Model (SVR): The SVR model consistently outperformed the other models, particularly when using multivariate inputs. It was also the fastest to train, making it a practical
  choice for real-time BGL prediction. The TML model's ability to integrate additional data like insulin dosage and physical activity proved beneficial in improving prediction
  accuracy, particularly for longer prediction horizons (60 minutes).
* DNN Model (LSTM): While the DNN model showed promising results, its performance was not significantly better than the TML model. It was also the slowest to train, which may limit
  its practicality for real-time applications. Interestingly, the DNN model performed similarly whether using univariate or multivariate inputs, suggesting that it might not fully
  utilize the additional data as effectively as the TML model.

## Input Comparison

The comparison of univariate and multivariate inputs yielded mixed results.

* Univariate Input: Using only BG data for prediction performed well for all models, especially for short-term predictions (30 minutes). This is consistent with previous research
  suggesting that continuous glucose monitoring (CGM) data alone is sufficient for practical BGL prediction in real-world settings.
* Multivariate Input: While the ML model benefited from the inclusion of additional variables (carbohydrates, insulin, physical activity), the other models did not show significant
  improvements. In fact, adding multivariate data sometimes degraded the performance of the ARIMA model. The DNN model’s performance remained largely unchanged, regardless of input
  type, implying that more advanced techniques for integrating multivariate data might be necessary to fully exploit its potential.

## Discussion

The authors of the study highlight several important findings. First, ML models (particularly SVR) are highly effective for BG prediction, especially when additional data is
available. Second, while multivariate input can improve prediction performance, especially for ML models, simply adding more variables does not guarantee better results. Advanced
data fusion methods may be required to fully utilize multivariate inputs.

The study also emphasizes the practical implications of model selection. TML models, which are faster to train and perform well with multivariate data, could be more suitable for
real-time applications, such as automated insulin delivery systems or continuous glucose monitoring devices.

## Conclusion

The comprehensive analysis provided by this study offers valuable insights into the performance of different data-driven models for blood glucose prediction in T1DM. The findings
suggest that ML models, particularly those using multivariate inputs, may offer the best balance of accuracy and speed for real-time BGL prediction. However, further research is
needed to explore advanced techniques for integrating multivariate data, as well as the potential of hybrid models that combine data-driven and physiological approaches.

