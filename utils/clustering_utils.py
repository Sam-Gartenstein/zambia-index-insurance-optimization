#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd  
import numpy as np 
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt
import geopandas as gpd
import folium


def apply_kmeans_clustering(dataframe, reduced_features, n_clusters, n_init=25, column_name="non_opt_cluster"):
    """
    Applies K-Means clustering to the given reduced feature set and assigns cluster labels to the original dataframe.

    Parameters:
    - dataframe (pd.DataFrame): The original DataFrame to which cluster labels will be added.
    - reduced_features (np.array or pd.DataFrame): The feature set used for clustering (e.g., PCA-transformed features).
    - n_clusters (int): Number of clusters for K-Means.
    - n_init (int, optional): Number of times the K-Means algorithm will run with different centroid seeds (default: 25).
    - column_name (str, optional): Name of the column to store cluster labels in the original DataFrame (default: "non_opt_cluster").

    Returns:
    - pd.DataFrame: The updated DataFrame with the assigned cluster labels.
    """
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    kmeans.fit(reduced_features)
    
    # Add cluster labels to the original DataFrame
    dataframe[column_name] = kmeans.labels_
    
    return dataframe


def plot_clustered_geodataframe(geodataframe, cluster_column="non_opt_cluster", title="K-Means Clusters on Seasonal Precipitation Data", cmap="viridis"):
    """
    Plots a GeoDataFrame with clusters visualized using a color map.

    Parameters:
    - geodataframe (gpd.GeoDataFrame): The GeoDataFrame containing the clustered data.
    - cluster_column (str, optional): The column in the GeoDataFrame representing cluster labels (default: "non_opt_cluster").
    - title (str, optional): The title of the plot (default: "K-Means Clusters on Seasonal Precipitation Data").
    - cmap (str, optional): The colormap used for visualization (default: "viridis").

    Returns:
    - None: Displays the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Check if geometry column exists
    if "geometry" not in geodataframe.columns:
        raise ValueError("GeoDataFrame must contain a 'geometry' column.")

    # Plotting clusters
    geodataframe.plot(column=cluster_column, cmap=cmap, legend=True, alpha=0.8, markersize=10, ax=ax)

    # Add title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Show plot
    plt.show()


def plot_folium_cluster_map(geodataframe, cluster_column="non_opt_cluster", zoom_start=6, tiles="cartodb positron"):
    """
    Plots a Folium map with cluster-based colored markers for a GeoDataFrame.

    Parameters:
    - geodataframe (gpd.GeoDataFrame): The GeoDataFrame containing the clustered data.
    - cluster_column (str, optional): The column in the GeoDataFrame representing cluster labels (default: "non_opt_cluster").
    - zoom_start (int, optional): Initial zoom level for the Folium map (default: 6).
    - tiles (str, optional): Tile style for the Folium map (default: "cartodb positron").

    Returns:
    - folium.Map: The generated map object.
    """
    # Ensure the cluster column exists
    if cluster_column not in geodataframe.columns:
        raise ValueError(f"The '{cluster_column}' column is missing. Ensure clustering was performed before mapping.")

    # Get the number of unique clusters
    num_clusters = geodataframe[cluster_column].nunique()

    # Generate distinct colors for clusters
    colormap = plt.cm.get_cmap('tab20', num_clusters)  # Uses 'tab20' for distinct colors
    colors = {cluster: f'#{int(colormap(i)[0]*255):02x}{int(colormap(i)[1]*255):02x}{int(colormap(i)[2]*255):02x}'
              for i, cluster in enumerate(sorted(geodataframe[cluster_column].unique()))}

    # Calculate map center
    map_center = [geodataframe.geometry.y.mean(), geodataframe.geometry.x.mean()]
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=tiles)

    # Add points to the map
    for _, row in geodataframe.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=colors[row[cluster_column]],  # Assigns color based on cluster
            fill=True,
            fill_color=colors[row[cluster_column]],
            fill_opacity=0.7,
            popup=f"<b>Cluster:</b> {row[cluster_column]}<br><b>Pixel ID:</b> {row['Pixel_ID']}"
        ).add_to(m)

    return m



