
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter


'''
Add in mean function
'''

def compute_cluster_mean_precipitation(dataframe, cluster_column="non_opt_cluster", date_column="agricultural_season", value_column="precipitation"):
    """
    Groups the given DataFrame by cluster and date, computing the mean precipitation for each cluster per month.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing precipitation data.
    - cluster_column (str, optional): The column representing cluster labels (default: "non_opt_cluster").
    - date_column (str, optional): The column representing the time period (default: "year_month").
    - value_column (str, optional): The column containing precipitation values (default: "precipitation").

    Returns:
    - pd.DataFrame: A DataFrame with mean precipitation per cluster per month.
    """
    grouped_df = dataframe.groupby([cluster_column, date_column], as_index=False)[value_column].mean()
    return grouped_df


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


def count_unique_drought_years(df, admin_col="ADM1_NAME"):
    """
    Counts the number of drought years per administrative region and returns
    the number of unique drought year counts.

    Parameters:
    - df (DataFrame): Input DataFrame that includes 'drought_indicator'.
    - admin_col (str): Column name for the administrative unit to group by.

    Returns:
    - ndarray: Unique drought year counts across the regions.
    """
    drought_counts = (
        df[df["drought_indicator"] == 1]
        .groupby(admin_col)["drought_indicator"]
        .count()
        .reset_index(name="drought_years")
    )

    return print(drought_counts["drought_years"].unique())


def plot_drought_heatmap_interactive(dataframe, cluster_column="non_opt_cluster", season_column="agricultural_season", drought_column="drought_indicator", precip_column="precipitation"):
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
    

def plot_drought_heatmap_static(
    dataframe,
    cluster_column="non_opt_cluster",
    season_column="agricultural_season",
    drought_column="drought_indicator",
    cmap="RdYlGn_r"
):
    """
    Creates a static heatmap of drought indicators for each cluster across agricultural seasons.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing drought data.
    - cluster_column (str, optional): The column representing clusters (default: "non_opt_cluster").
    - season_column (str, optional): The column representing agricultural seasons (default: "agricultural_season").
    - drought_column (str, optional): The column representing drought indicators (default: "drought_indicator").
    - cmap (str, optional): Colormap for the heatmap (default: "RdYlGn_r", where green = non-drought, red = drought).

    Returns:
    - None: Displays a static heatmap.
    """

    # Pivot the data for heatmap structure
    heatmap_data = dataframe.pivot(index=cluster_column, columns=season_column, values=drought_column)

    # Create the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        cmap=["green", "red"],   # Explicit colors for 0 and 1
        cbar_kws={"ticks": [0, 1], "label": "Drought Indicator"},
        linewidths=0.5,
        linecolor="white",
        annot=False
    )

    # Formatting
    plt.title("Drought Indicator Heatmap by Cluster")
    plt.xlabel("Agricultural Season (Years)")
    plt.ylabel("Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
    
    
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


def plot_drought_percentage_heatmap_interactive(drought_summary, cluster_column="non_opt_cluster", season_column="agricultural_season", drought_percentage_column="drought_percentage"):
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


def plot_drought_percentage_heatmap_static(
    drought_summary: pd.DataFrame,
    cluster_column: str = "non_opt_cluster",
    season_column: str = "agricultural_season",
    drought_percentage_column: str = "drought_percentage",
    *,
    title: str = "Drought Percentage by Cluster and Agricultural Season",
    annotate: bool = False,
    vmin: float = 0.0,
    vmax: float = 100.0,
):
    """
    Draw a static heatmap of drought percentage by cluster (rows) and season (columns).

    Parameters
    ----------
    drought_summary : pd.DataFrame
        DataFrame with at least [cluster_column, season_column, drought_percentage_column].
    cluster_column : str
        Column name for clusters (rows).
    season_column : str
        Column name for seasons/years (columns).
    drought_percentage_column : str
        Column name for drought percentage values (0–100).
    title : str
        Plot title.
    annotate : bool
        If True, write the percentage in each cell.
    vmin, vmax : float
        Color scale bounds (defaults to 0–100).

    Returns
    -------
    None (shows the figure)
    """
    # Pivot to wide matrix: rows=clusters, cols=seasons
    heatmap_df = drought_summary.pivot(
        index=cluster_column, columns=season_column, values=drought_percentage_column
    )

    # Sort the axes for consistent ordering
    heatmap_df = heatmap_df.sort_index().reindex(sorted(heatmap_df.columns), axis=1)

    # Convert to array
    Z = heatmap_df.to_numpy(dtype=float)

    # Colormap: reverse Red-Yellow-Green (higher % = red), gray for NaNs
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#e0e0e0")  # light gray for missing cells

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Axis ticks/labels
    ax.set_title(title)
    ax.set_xlabel("Agricultural Season")
    ax.set_ylabel("Cluster")

    seasons = list(heatmap_df.columns)
    clusters = list(heatmap_df.index)

    ax.set_xticks(np.arange(len(seasons)))
    ax.set_xticklabels([str(x) for x in seasons], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_yticklabels([str(x) for x in clusters])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Drought Percentage (%)")

    # Optional annotations
    if annotate:
        nrows, ncols = Z.shape
        for r in range(nrows):
            for c in range(ncols):
                val = Z[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.0f}%", ha="center", va="center", fontsize=8)

    # Light gridlines between cells (via minor ticks)
    ax.set_xticks(np.arange(-0.5, Z.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, Z.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.show()
    
    
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


def plot_forgone_by_zones(
    zones_to_fp: dict,
    title: str = "Forgone Payments by Number of Zones",
    annotate_in_millions: bool = True,
    x_label_template: str = "{zones} zones",
    *,
    figsize=(8, 5),
    line_kwargs=None,                 # e.g., {"marker":"o","linestyle":"-"}
    annotate: bool = True,
    annotate_fmt=None,                # callable: float -> str; overrides annotate_in_millions if provided
    annotate_offsets=None,            # None | (dx,dy) | list[(dx,dy)] | callable(index,value)->(dx,dy)
    annotate_ha: str = "center",
    annotate_va: str = "bottom",
    show: bool = True,
):
    """
    Plot forgone payments vs. number of zones, with configurable annotation positions.

    Parameters
    ----------
    zones_to_fp : dict[int, float]
        Mapping {num_zones: forgone_payments}.
    title : str
        Plot title.
    annotate_in_millions : bool
        If True, annotate each point as '<value>M' with 2 decimals.
        Otherwise annotate as raw value with thousands separators.
        Ignored if `annotate_fmt` is provided.
    x_label_template : str
        Template for x tick labels. Use '{zones}' placeholder (e.g., '{zones} zones').
    figsize : tuple
        Figure size.
    line_kwargs : dict
        Passed to plt.plot (e.g., marker, linestyle).
    annotate : bool
        Toggle annotations on/off.
    annotate_fmt : callable or None
        Custom formatter for annotation text. Signature: f(value)->str.
    annotate_offsets : None | (dx,dy) | list[(dx,dy)] | callable
        Offset(s) in points for annotation placement.
        - None: use (0, 8) for all points
        - (dx,dy): apply to all points
        - list of (dx,dy): per-point offsets, must match number of points
        - callable: given (i, y) returns (dx, dy) for point i with value y
    annotate_ha, annotate_va : str
        Horizontal/vertical alignment for annotation text.
    show : bool
        If True, calls plt.show(). Returns (fig, ax) regardless.

    Returns
    -------
    (fig, ax)
    """
    # Sort by zone count for consistent x-ordering
    pairs = sorted(zones_to_fp.items(), key=lambda kv: kv[0])
    zones  = [z for z, _ in pairs]
    values = [v for _, v in pairs]
    x_labels = [x_label_template.format(zones=z) for z in zones]

    # Defaults
    _line_kwargs = dict(marker="o", linestyle="-")
    if line_kwargs:
        _line_kwargs.update(line_kwargs)

    # Default annotation formatter (no currency)
    if annotate_fmt is None:
        if annotate_in_millions:
            annotate_fmt = lambda v: f"{v/1e6:.2f}M"
        else:
            annotate_fmt = lambda v: f"{v:,.0f}"

    # Build figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_labels, values, **_line_kwargs)

    ax.set_xlabel("Number of zones")
    ax.set_ylabel("Forgone Payments")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    # Resolve annotation offsets
    def resolve_offset(i, y):
        if annotate_offsets is None:
            return (0, 8)
        if isinstance(annotate_offsets, tuple):
            return annotate_offsets
        if callable(annotate_offsets):
            return annotate_offsets(i, y)
        # assume list-like
        return annotate_offsets[i]

    # Annotate points
    if annotate:
        n = len(values)
        if isinstance(annotate_offsets, list) and len(annotate_offsets) != n:
            raise ValueError("Length of annotate_offsets list must match number of points.")
        for i, (xl, y) in enumerate(zip(x_labels, values)):
            if y is None:
                continue
            dx, dy = resolve_offset(i, y)
            ax.annotate(
                annotate_fmt(y),
                xy=(xl, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=annotate_ha,
                va=annotate_va,
                fontsize=10,
            )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax

DEFAULT_LABELS = ["Admin 1 → Clusters", "Clusters → Admin 2", "Admin 2 → Camps"]

def benefit_changes(
    values,
    *,
    labels=DEFAULT_LABELS,
    title="Marginal Benefit per Zone Change",
    ylabel="Absolute change per added zone (ZMW per zone)",
    figsize=(8, 5),
    yformatter=lambda v: f"{v:,.0f}",
    line_kwargs=None,
    annotate=True,
    annotate_fmt=lambda v: f"{v:,.2f}",
    annotate_offsets=None,   # None, (dx,dy), or list of (dx,dy) per point (in points)
    annotate_ha="center",
    annotate_va="bottom",
):
    """
    Plot marginal change per added zone with configurable title and annotation positions.

    Parameters
    ----------
    values : array-like of float
        Y values (e.g., marginal change, ZMW/zone).
    labels : list[str], optional
        X tick labels. Defaults to Admin→Clusters→Admin2→Camps.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, optional
        Figure size.
    yformatter : callable, optional
        Formats Y tick labels. Default: thousands separators, no decimals.
    line_kwargs : dict, optional
        Extra kwargs for ax.plot (e.g., {"marker":"o","linestyle":"-"}).
    annotate : bool, optional
        If True, annotate each point with its value.
    annotate_fmt : callable, optional
        Formats annotation text from numeric value.
    annotate_offsets : None | (dx,dy) | list[(dx,dy)], optional
        Offset(s) for annotations in points. If None → (0,8) for all.
        If tuple → applied to all points. If list → per-point offsets.
    annotate_ha : {"left","center","right"}, optional
        Horizontal alignment for annotations.
    annotate_va : {"top","bottom","center","baseline","center_baseline"}, optional
        Vertical alignment for annotations.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    values = np.asarray(values, dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot line
    _line_kwargs = dict(marker="o", linestyle="-")
    if line_kwargs:
        _line_kwargs.update(line_kwargs)
    ax.plot(x, values, **_line_kwargs)

    # Axes formatting
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: yformatter(v)))

    # Annotation offsets
    if annotate_offsets is None:
        offsets = [(0, 8)] * len(values)
    elif isinstance(annotate_offsets, tuple):
        offsets = [annotate_offsets] * len(values)
    else:
        # assume list of per-point offsets
        offsets = annotate_offsets
        if len(offsets) != len(values):
            raise ValueError("Length of annotate_offsets must match number of values.")

    # Annotate points
    if annotate:
        for (xi, yi), (dx, dy) in zip(zip(x, values), offsets):
            if np.isfinite(yi):
                ax.annotate(
                    annotate_fmt(yi),
                    xy=(xi, yi),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=annotate_ha,
                    va=annotate_va,
                )

    plt.tight_layout()
    return fig, ax


