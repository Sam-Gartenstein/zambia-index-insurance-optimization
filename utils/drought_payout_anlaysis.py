
import pandas as pd
import plotly.express as px
import numpy as np


'''
Add in mean function
'''

def assign_drought_indicator(dataframe, group_column="non_opt_cluster", season_column="agricultural_season", precip_column="precipitation", drought_column="drought_indicator", quantile_threshold=0.2):
    """
    Assigns a drought indicator to each observation based on a column-specific precipitation threshold.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing precipitation data.
    - group_column (str, optional): The column to group by (default: "non_opt_cluster").
    - season_column (str, optional): The column representing agricultural seasons (default: "agricultural_season").
    - precip_column (str, optional): The column representing precipitation values (default: "precipitation").
    - drought_column (str, optional): The name of the new column to store drought indicators (default: "drought_indicator").
    - quantile_threshold (float, optional): The percentile threshold for defining drought (default: 0.2, i.e., 20th percentile).

    Returns:
    - pd.DataFrame: The DataFrame with an added drought indicator column.
    """
    # Compute the drought threshold for each group
    drought_thresholds = dataframe.groupby(group_column)[precip_column].quantile(quantile_threshold)

    # Assign drought indicator (1 = drought, 0 = non-drought)
    dataframe[drought_column] = dataframe.apply(
        lambda row: 1 if row[precip_column] <= drought_thresholds[row[group_column]] else 0, axis=1
    )

    return dataframe


def plot_drought_heatmap(dataframe, cluster_column="non_opt_cluster", season_column="agricultural_season", drought_column="drought_indicator", precip_column="precipitation"):
    """
    Creates an interactive heatmap of drought indicators for each cluster across agricultural seasons.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing drought data.
    - cluster_column (str, optional): The column representing clusters (default: "non_opt_cluster").
    - season_column (str, optional): The column representing agricultural seasons (default: "agricultural_season").
    - drought_column (str, optional): The column representing drought indicators (default: "drought_indicator").
    - precip_column (str, optional): The column representing precipitation values (default: "precipitation").

    Returns:
    - None: Displays an interactive heatmap.
    """

    # Pivot the data for heatmap structure
    heatmap_data = dataframe.pivot(index=cluster_column, columns=season_column, values=drought_column)
    precip_data = dataframe.pivot(index=cluster_column, columns=season_column, values=precip_column)

    # Create hover text annotations
    hover_text = pd.DataFrame(
        [
            [
                f"Cluster: {cluster}<br>Season: {year}<br>Drought: {'Yes' if heatmap_data.loc[cluster, year] == 1 else 'No'}<br>Precipitation: {precip_data.loc[cluster, year]:.2f} mm"
                if pd.notna(heatmap_data.loc[cluster, year]) else ""
                for year in heatmap_data.columns
            ]
            for cluster in heatmap_data.index
        ],
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )

    # Create the heatmap
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale=["green", "red"],  # Green for non-drought, Red for drought
        labels={"x": "Agricultural Season", "y": "Cluster", "color": "Drought Indicator"},
    )

    # Add hover text with precipitation
    fig.update_traces(
        customdata=hover_text.values,  # Attach hover text data
        hovertemplate="%{customdata}<extra></extra>",  # Show text without extra trace name
        xgap=2,  # Add spacing between cells for visual clarity
        ygap=2
    )

    # Ensure proper year labeling
    years = sorted(heatmap_data.columns)
    fig.update_layout(
        title="Drought Indicator Heatmap by Cluster",
        xaxis_title="Agricultural Season (Years)",
        yaxis_title="Cluster",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(years))),
            ticktext=[str(year) for year in years]
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(heatmap_data.index))),
            ticktext=[str(cluster) for cluster in heatmap_data.index]
        ),
        coloraxis_colorbar=dict(
            title="Drought Indicator",
            tickvals=[0, 1],
            ticktext=["Non-Drought", "Drought"]
        ),
        plot_bgcolor="white"
    )

    # Show the heatmap
    fig.show()
    

def summarize_drought_by_cluster(dataframe, cluster_column="non_opt_cluster", season_column="agricultural_season", pixel_column="Pixel_ID", drought_column="drought_indicator"):
    """
    Groups the data by cluster and agricultural season to compute the total pixels, drought pixels, 
    and the percentage of pixels in drought.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing drought data.
    - cluster_column (str, optional): The column representing clusters (default: "non_opt_cluster").
    - season_column (str, optional): The column representing agricultural seasons (default: "agricultural_season").
    - pixel_column (str, optional): The column representing pixel IDs (default: "Pixel_ID").
    - drought_column (str, optional): The column representing drought indicators (default: "drought_indicator").

    Returns:
    - pd.DataFrame: A summary DataFrame with total pixels, drought pixels, and drought percentage.
    """
    drought_summary = (
        dataframe
        .groupby([cluster_column, season_column])
        .agg(
            total_pixels=(pixel_column, "count"),  # Count total pixels
            drought_pixels=(drought_column, "sum")  # Sum of drought pixels
        )
        .reset_index()
    )

    # Calculate the percentage of pixels in drought
    drought_summary["drought_percentage"] = (drought_summary["drought_pixels"] / drought_summary["total_pixels"]) * 100

    return drought_summary


def plot_drought_percentage_heatmap(drought_summary, cluster_column="non_opt_cluster", season_column="agricultural_season", drought_percentage_column="drought_percentage"):
    """
    Creates an interactive heatmap showing drought percentage by cluster and agricultural season.

    Parameters:
    - drought_summary (pd.DataFrame): The summary DataFrame containing drought percentage data.
    - cluster_column (str, optional): The column representing clusters (default: "non_opt_cluster").
    - season_column (str, optional): The column representing agricultural seasons (default: "agricultural_season").
    - drought_percentage_column (str, optional): The column representing drought percentages (default: "drought_percentage").

    Returns:
    - None: Displays an interactive heatmap.
    """

    # Pivot the data for heatmap structure
    heatmap_data = drought_summary.pivot(index=cluster_column, columns=season_column, values=drought_percentage_column)

    # Create the heatmap
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale="RdYlGn_r",  # Reverse Red-Yellow-Green for higher drought percentages in red
        labels={"x": "Agricultural Season", "y": "Cluster", "color": "Drought Percentage (%)"},
    )

    # Update layout for readability
    fig.update_layout(
        title="Drought Percentage by Cluster and Agricultural Season",
        xaxis_title="Agricultural Season",
        yaxis_title="Cluster",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(drought_summary[season_column].unique()))),
            ticktext=[str(year) for year in sorted(drought_summary[season_column].unique())],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(drought_summary[cluster_column].unique()))),
            ticktext=[str(cluster) for cluster in sorted(drought_summary[cluster_column].unique())],
        ),
        coloraxis_colorbar=dict(title="Drought %"),
    )

    # Show the plot
    fig.show()


def compute_payout_balance_index(df, drought_percentage_column="drought_percentage"):
    """
    Adds 'deviation_from_50' and 'payout_balance_index' columns to the DataFrame.

    The payout balance index is calculated as:
        PBI = (1 - ((50 - deviation_from_50) / 50)) * 100
            where deviation_from_50 = |P - 50|

    This rewards cohesive clusters (0% or 100% receiving payouts) with high PBI scores.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing drought payout percentages.

    drought_percentage_column : str
        Column containing the % of pixels receiving payouts (0–100 scale).

    Returns:
    --------
    pd.DataFrame
        A copy of the original DataFrame with:
            - 'deviation_from_50'
            - 'payout_balance_index'
    """
    df = df.copy()
    df["deviation_from_50"] = np.abs(df[drought_percentage_column] - 50)
    df["payout_balance_index"] = (1 - ((50 - df["deviation_from_50"]) / 50)) * 100
    return df


def filter_low_drought(df):
    """
    Filters the input DataFrame to include only rows where drought_percentage is equal to or greater than 50%.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a 'drought_percentage' column.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    return df[df["drought_percentage"] >= 50]
