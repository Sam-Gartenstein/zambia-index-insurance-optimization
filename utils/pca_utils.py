#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go

def apply_minmax_scaler_rowwise(numeric_data):
    scaler = MinMaxScaler()
    scaled_numeric_array = np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in numeric_data])
    return scaled_numeric_array


def perform_pca_analysis(scaled_numeric_array):
    # Perform PCA directly on the scaled row-wise data
    pca = PCA()
    pca_transformed = pca.fit_transform(scaled_numeric_array)

    # Compute explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    return pca_transformed, explained_variance, cumulative_variance


def plot_cumulative_variance(cumulative_variance):
    # Create an interactive plot for cumulative variance
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_variance) + 1)),
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Variance',
        marker=dict(size=8),
        line=dict(color='green')
    ))

    # Update layout
    fig.update_layout(
        title="Cumulative Explained Variance (PCA)",
        xaxis_title="Principal Component",
        yaxis_title="Cumulative Variance Explained",
        hovermode="x unified"
    )

    # Show interactive plot
    fig.show()

