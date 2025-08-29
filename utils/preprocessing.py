import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point
import ee
import geemap


def summarize_admin_level(admin_level: str, name_field: str):
    """
    Load GAUL data for a given admin level, print unit names,
    and print the number of unique units.

    Parameters
    ----------
    admin_level : str
        'level1' or 'level2'
    name_field : str
        Column name for the admin level ('ADM1_NAME' or 'ADM2_NAME')
    """
    gdf, names = load_fao_gaul_data(admin_level=admin_level)

    print(f"Administrative {admin_level.title()} Units ({name_field}):")
    print(names, "\n")

    num_unique = gdf[name_field].nunique()
    print(f"Number of unique {name_field} units: {num_unique}\n")

    return gdf, names


def load_fao_gaul_data(admin_level="level2", country_name="Zambia"):
    """
    Load GAUL data from GEE for a given country and admin level, return as GeoDataFrame.

    Parameters:
        admin_level (str): 'level0', 'level1', or 'level2'
        country_name (str): e.g., 'Zambia'

    Returns:
        tuple: (GeoDataFrame, list of admin names)
    """
    try:
        fc = ee.FeatureCollection(f"FAO/GAUL/2015/{admin_level}")
        fc_filtered = fc.filter(ee.Filter.eq("ADM0_NAME", country_name))

        name_field = {
            "level0": "ADM0_NAME",
            "level1": "ADM1_NAME",
            "level2": "ADM2_NAME"
        }.get(admin_level, "ADM2_NAME")

        counties_list = fc_filtered.aggregate_array(name_field).getInfo()

        # Convert Earth Engine FeatureCollection to GeoDataFrame
        gdf = geemap.ee_to_gdf(fc_filtered)

        return gdf, counties_list
    except Exception as e:
        print(f"Error: {e}")
        return None, []


def subset_precip_data(precip_xr, time_ranges):
    """
    Efficiently subsets an xarray Dataset for multiple time ranges using dictionary comprehension.

    Parameters:
    - precip_xr (xarray.Dataset): Full precipitation dataset.
    - time_ranges (dict): Dictionary with labels as keys and (start, end) date tuples as values.

    Returns:
    - dict: Dictionary where keys are time labels (e.g., "8185") and values are subsetted xarray datasets.
    """
    return {label: precip_xr.sel(time=slice(start, end)) for label, (start, end) in time_ranges.items()}


def pivot_precip_data(precip_xr_subset):
    """
    Converts an xarray Dataset of precipitation data into a pivoted DataFrame 
    where each (lat, lon) pair is a row and each time step is a column.

    Parameters:
    - precip_xr_subset (xarray.Dataset): A subset of the precipitation dataset with selected time range.

    Returns:
    - pd.DataFrame: Pivoted DataFrame with (lat, lon) as index and time steps as columns.
    """
    df = precip_xr_subset.to_dataframe().reset_index()
    pivot_dict = {}
    unique_coords = df.groupby(["lat", "lon"])

    for (lat, lon), group in tqdm(unique_coords, desc="Pivoting Data", unit="grid point"):
        pivot_dict[(lat, lon)] = group.set_index("time")["precipitation"].to_dict()

    df_pivot = pd.DataFrame.from_dict(pivot_dict, orient="index")
    df_pivot.index = pd.MultiIndex.from_tuples(df_pivot.index, names=["lat", "lon"])
    df_pivot.columns = [str(col) for col in df_pivot.columns]

    print(f"DataFrame shape after pivoting: {df_pivot.shape}")
    return df_pivot


def clean_and_geocode_pivot(df_pivot):
    """
    Cleans the pivoted precipitation DataFrame by removing NaNs, resetting the index,
    adding a geometry column for geospatial analysis, and converting it into a GeoDataFrame.

    Parameters:
    - df_pivot (pd.DataFrame): Pivoted DataFrame with (lat, lon) as index and time steps as columns.

    Returns:
    - gpd.GeoDataFrame: Cleaned GeoDataFrame.
    """
    df_pivot_cleaned = df_pivot.dropna()
    df_pivot_reset = df_pivot_cleaned.reset_index()

    print(f"DataFrame shape after dropping NaNs: {df_pivot_reset.shape}")

    df_pivot_reset["geometry"] = df_pivot_reset.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    df_pivot_reset.insert(0, "Pixel_ID", ["Pixel " + str(i+1) for i in range(len(df_pivot_reset))])

    gdf_pixels = gpd.GeoDataFrame(df_pivot_reset, geometry="geometry", crs="EPSG:4326")
    gdf_pixels = gdf_pixels.loc[:, ~gdf_pixels.columns.duplicated()]

    column_order = ["Pixel_ID", "lat", "lon", "geometry"] + [
        col for col in gdf_pixels.columns if col not in ["Pixel_ID", "lat", "lon", "geometry"]
    ]
    gdf_pixels = gdf_pixels[column_order]

    print(f"GeoDataFrame shape after conversion: {gdf_pixels.shape}")

    return gdf_pixels


def spatial_join_by_intersects(precip_gdf, admin_gdf, admin_col='ADM1_NAME', epsg=4326):
    """
    Ensure both GeoDataFrames are in the same CRS and perform a spatial join 
    based on the 'intersects' predicate. Drops the 'index_right' column after joining.

    Parameters:
    - precip_gdf (GeoDataFrame): GeoDataFrame with point or polygon data (e.g., precipitation data).
    - admin_gdf (GeoDataFrame): GeoDataFrame with administrative boundaries.
    - admin_col (str): The column name in admin_gdf to keep in the join.
    - epsg (int): The EPSG code for the target CRS (default is 4326).

    Returns:
    - GeoDataFrame: Result of the spatial join with the admin column attached, excluding 'index_right'.
    """
    precip_gdf = precip_gdf.to_crs(epsg=epsg)
    admin_gdf = admin_gdf.to_crs(epsg=epsg)

    # Filter invalid geometries (optional but recommended)
    precip_gdf = precip_gdf[precip_gdf.is_valid]
    admin_gdf = admin_gdf[admin_gdf.is_valid]

    joined = gpd.sjoin(
        precip_gdf,
        admin_gdf[[admin_col, 'geometry']],
        how='left',
        predicate='intersects'
    )

    if 'index_right' in joined.columns:
        joined = joined.drop(columns='index_right')

    return joined


def long_format_precipitation(df, cluster_columns=None):
    """
    Converts a wide-format precipitation DataFrame into long format,
    where each row corresponds to a single observation of precipitation
    for a given pixel and agricultural season.

    Parameters:
    -----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        A DataFrame with columns including 'Pixel_ID', 'lat', 'lon', 'geometry',
        and one column per agricultural season (e.g., years like 1982, 1983, ..., 2024),
        containing precipitation values. May also include additional metadata columns.

    cluster_columns : list of str, optional
        List of additional columns (e.g., cluster labels) to retain in the melted output.

    Returns:
    --------
    pandas.DataFrame
        A long-format DataFrame with columns:
        'Pixel_ID', 'lat', 'lon', 'geometry', optionally cluster_columns,
        'agricultural_season' (as int), and 'precipitation'.
    """
    # Base columns to retain
    id_vars = ['Pixel_ID', 'lat', 'lon', 'geometry']
    
    # Add any additional columns to retain
    if cluster_columns:
        id_vars += cluster_columns

    # Identify value columns (assumed to be agricultural season years)
    value_vars = [col for col in df.columns if col not in id_vars]

    # Melt the DataFrame
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='agricultural_season',
        value_name='precipitation'
    )

    # Convert agricultural_season to int
    df_long['agricultural_season'] = df_long['agricultural_season'].astype(int)

    return df_long


def compute_mean_precip(df, group_cols, value_col="precipitation"):
    """
    Groups the DataFrame by specified columns and computes the mean of the value column.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - group_cols (list): List of columns to group by.
    - value_col (str): Name of the column to average.

    Returns:
    - DataFrame with group means.
    """
    return df.groupby(group_cols)[value_col].mean().reset_index()
