
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from aif360.datasets import AdultDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.metrics import BinaryLabelDatasetMetric

st.set_page_config(layout="wide")
st.title("Bias Detection & Mitigation Tool (AIF360 + Adversarial Debiasing)")
st.write("This app uses the Adult dataset to demonstrate bias detection and mitigation using the AIF360 library.")

# Load the dataset
dataset = AdultDataset()
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

# Bias metric before mitigation
metric_orig = BinaryLabelDatasetMetric(dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
spd_before = metric_orig.statistical_parity_difference()

# Show sample data
st.subheader("Sample Data")
df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
st.dataframe(df.sample(5))

# Adversarial Debiasing model
sess = tf.Session()
ad = AdversarialDebiasing(privileged_groups=privileged_groups,
                          unprivileged_groups=unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess)
ad.fit(dataset)
dataset_transf = ad.predict(dataset)

# Bias metric after mitigation
metric_transf = BinaryLabelDatasetMetric(dataset_transf, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
spd_after = metric_transf.statistical_parity_difference()

# Show metrics
st.markdown(f"**Statistical Parity Difference (Before Mitigation):** 0. **{spd_before:.4f}**")
st.markdown(f"**Statistical Parity Difference (After Mitigation):** 0. **{spd_after:.4f}**")

if abs(spd_after) < abs(spd_before):
    st.success("✅ Bias successfully mitigated!")
else:
    st.error("⚠️ Bias mitigation was not effective.")
