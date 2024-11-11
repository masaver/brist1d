import numpy as np
import pandas as pd
import shap


class ShapWrapper:
    def __init__(self, model, X):
        self.model = model
        self.X = X
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)
        self.expected_value = self.explainer.expected_value

    def get_top_features(self, n: int = 10):
        feature_importances = pd.DataFrame({
            'feature': self.X.columns,
            'mean_abs_shap_value': np.abs(self.shap_values).mean(axis=0)
        }).sort_values(by='mean_abs_shap_value', ascending=False)

        return feature_importances['feature'].head(n).tolist()

    def get_shap_values(self):
        return self.shap_values

    def get_shap_values_df(self):
        return pd.DataFrame(self.shap_values, columns=self.X.columns)

    def get_shap_summary_plot(self):
        return shap.summary_plot(self.shap_values, self.X)

    def get_shap_dependence_plot(self, feature_name):
        return shap.dependence_plot(feature_name, self.shap_values, self.X)

    def get_shap_decision_plot(self, index):
        return shap.decision_plot(self.expected_value, self.shap_values[index], self.X.iloc[index])

    def get_shap_summary_plot_bar(self):
        return shap.summary_plot(self.shap_values, self.X, plot_type='bar')

    def get_shap_dependence_plot_bar(self, feature_name):
        return shap.dependence_plot(feature_name, self.shap_values, self.X, interaction_index=None, show=False)

    def get_shap_force_plot_bar(self, index):
        return shap.force_plot(self.expected_value, self.shap_values[index], self.X.iloc[index], show=False, matplotlib=True, text_rotation=0)

    def get_shap_decision_plot_bar(self, index):
        return shap.decision_plot(self.expected_value, self.shap_values[index], self.X.iloc[index], show=False)

    def get_shap_interaction_values(self, feature_name):
        return self.explainer.shap_interaction_values(self.X)[feature_name]

    def get_shap_interaction_plot(self, feature_name):
        return shap.summary_plot(self.get_shap_interaction_values(feature_name), self.X)

    def get_shap_interaction_plot_bar(self, feature_name):
        return shap.summary_plot(self.get_shap_interaction_values(feature_name), self.X, plot_type='bar')
