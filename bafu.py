# import contextily as ctx # UNTIL CONTEXTILY IS INSTALLED
import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
import shapely
# import time
# from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_columns', 100)
import warnings


def pp(df, n=3):
    tp = type(df).__name__
    srs = False
    geom_col_name = False
    if tp == 'GeoDataFrame':
        srs = df.crs.srs if df.crs is not None else 'no crs'
        try:
            geom_col_name = df.geometry.name
        except AttributeError as e:
            geom_col_name = '!missing'
    print(*(el for el in ['('+ ', '.join(tuple(f'{el:_}' for el in df.shape)).replace('_', '`') +')',
                          tp, srs, geom_col_name] if el), sep=' | ')
    display(df.head(n))
    
    
# UNTIL CONTEXTILY IS INSTALLED
# def drw(gdf_list, *args, figsize=(8,8), **kwargs):
#     '''
#     Draw multiple GeoDataFrames on same ax. Pass plot kwargs either as parameters
#     (if params are the same for all gdf's or there is only one gdf)
#     or as dict list [dict(par='val1'), dict(par='val2')] etc.
#     '''
#     if type(gdf_list) != list:
#         gdf_list = [gdf_list]
#     fig, ax = plt.subplots(1,1, figsize=figsize)
#     for idx, gdf in enumerate(gdf_list):
#         if args:  # if we have a list of dicts
#             gdf.to_crs(3857).plot(ax=ax, **args[0][idx])
#         elif kwargs:  # if we have kwargs - one set of params
#             if len(gdf_list) > 1:  # but more than 1 gdf - apply same set to all gdf's
#                 gdf.to_crs(3857).plot(ax=ax, **kwargs)
#         else:  # if we have no additional parameters
#             gdf.to_crs(3857).plot(ax=ax)
#     ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
#     ax.set_axis_off()
    
    
def construct_gdf(df, crs=4326, geom_col='geometry', wkb=True, **kwargs):
    if 'xy' in kwargs:  # x-y geometry is specified
        return gpd.GeoDataFrame(df, crs=crs, geometry=gpd.GeoSeries.from_xy(df[kwargs['xy'][0]], df[kwargs['xy'][1]]))
    elif wkb == False:  # wkt geometry
        return gpd.GeoDataFrame(df, crs=crs, geometry=gpd.GeoSeries.from_wkt(df[geom_col]))
    else:  # wkb geometry
        return gpd.GeoDataFrame(df, crs=crs, geometry=gpd.GeoSeries.from_wkb(df[geom_col]))
    
    
def telegram(msg='default message', who='Gleb'):
    import requests
    from cred import tg
    chat_id = tg[who][0]
    tok = tg[who][1]
    url = f"https://api.telegram.org/bot{tok}/sendMessage?chat_id={chat_id}&text={msg}"
    requests.get(url)


def gdf_utm_code(gdf):
    import math
    '''returns epsg code from 4326 gdf'''
    
    def wgs_to_utm(lon: float, lat: float):
        '''given x and y in 4326 returns epsg code'''
        utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
        if len(utm_band) == 1:
            utm_band = '0' + utm_band
        if lat >= 0:
            return int('326' + utm_band)
        return int('327' + utm_band)
    
    x_min, y_min, x_max, y_max = gdf.to_crs(4326).total_bounds
    return wgs_to_utm((x_min+x_max)/2, (y_min+y_max)/2)


def fillna_nearest(gdf_na, col_dict, *args, max_distance=None, distance_col=None):
    '''
    Fills missing values of one GeoDataFrame from nearest features of another (could be the same one).
    
    Given gdf with nans, list(dict) of columns with nans, returns enriched gdf where in
    said columns nan values are replaced by nearest features from target gdf. If target
    gdf not specified - it will be the same input gdf with nans. If distance_col
    specified - will add distance to nearest notna feature that was used as a filler
    '''
    
    import warnings
    
    if not gdf_na.index.is_unique:  # if we have a non-unique index - result will be faulty
        raise IndexError('geodataframe w/nans\' index is not unique')
    if gdf_na.crs.srs == 'epsg:4326':
        warnings.warn('\ngdf to fill uses epsg:4326 - distance calculations might be incorrect; consider using equidistant crs')
    if len(args) > 1:
        warnings.warn(f'\n{len(args)} positional arguments were given, expected max 1 - gdf with values to fill na\'s')
    if type(col_dict) == list:  # if we have list w/cols to fill - we assume columns have same name
        col_dict = {k: k for k in col_dict}
    if args:  # if we have explicitly specified gdf fill - we use it
        gdf_fill = args[0]
    else:  # otherwise we just take the same gdf with nans
        gdf_fill = gdf_na
    if gdf_na.crs != gdf_fill.crs:
        gdf_fill = gdf_fill.to_crs(gdf_na.crs)  # if crs mismatch - correct that
    gdf_out = gdf_na.copy()
    for key in col_dict:    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf_sjn = gdf_na[[gdf_na.geometry.name]].sjoin_nearest(
                gdf_fill[[gdf_fill.geometry.name, col_dict[key]]].dropna(subset=[col_dict[key]]),
                max_distance=max_distance, distance_col=distance_col, how='left').rename(
                columns={col_dict[key]: key})
        gdf_out = gdf_out.fillna(gdf_sjn)
        if distance_col is not None:
            gdf_out = gdf_out.join(gdf_sjn[[distance_col]].rename(
                columns={distance_col: f'{distance_col}_{key}'}), how='left')
    
    return gdf_out
    

def sjoin_lop_propsum(gdf_base, gdf_join, col_list):
    '''
    Enriches base gdf layer with features from join gdf based on overlap amount.
    
    Given two polygonal GeoDataFrames (base & join) and list of join columns,
    returns modified base gdf with stats for target column list (cols from join gdf):
    if column is numeric - returns proportional sum (portion of overlap area
    by numeric value of said column); otherwise - value of join gdf feature
    with largest overlap
    '''
    
    
    if not gdf_base.index.is_unique:  # if we have a non-unique index - result will be faulty
        raise IndexError('base geodataframe index is not unique')
    from pandas.api.types import is_numeric_dtype  # check if columns are numeric
    
    if gdf_join.crs != gdf_base.crs:
        gdf_join = gdf_join.to_crs(gdf_base.crs)  # if crs mismatch - correct that
    if gdf_base.crs.srs == 'epsg:4326':
        warnings.warn('\nbase gdf uses epsg:4326 - area and overlap calculations might be wrong; consider using area preserving crs')
    # gdf_join.geometry = gdf_join.geometry.apply(shapely.make_valid)
    # gdf_base.geometry = gdf_base.geometry.apply(shapely.make_valid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_join['join_geom_area'] = gdf_join.geometry.area  # original gdf_join area to compare to
        gdf_overlay = gpd.overlay(gdf_base.reset_index(), gdf_join.reset_index(), how='intersection')
        gdf_overlay['intersection_area'] = gdf_overlay.geometry.area
    
    gdf_out = gdf_base.copy()
    numeric_col_count = 0
    categorical_col_count = 0
    for col_name in col_list:
        if is_numeric_dtype(gdf_join[col_name]):  # if column we have is numeric - calculate proportional sum
            gdf_overlay[f'{col_name}_prsum'] = gdf_overlay[col_name] * gdf_overlay['intersection_area'] / gdf_overlay['join_geom_area']
            gdf_out = gdf_out.merge(gdf_overlay.groupby('index_1').sum()[f'{col_name}_prsum'],
                                    how='left', left_index=True, right_index=True)
            numeric_col_count += 1
        else:  # if column is categorical - get category of feature w/largest overlap
            gdf_out = gdf_out.merge(gdf_overlay.set_index(col_name)[['index_1', 'intersection_area']].groupby('index_1').idxmax(),
                                    how='left', left_index=True, right_index=True).rename(columns={'intersection_area': f'{col_name}_lop'})
            categorical_col_count += 1
        
    print(f'created gdf of shape {gdf_out.shape}, crs {gdf_out.crs.srs}, {numeric_col_count} num cols & {categorical_col_count} cat cols')
    return gdf_out[[col for col in gdf_out if col not in [gdf_out.geometry.name]] + [gdf_out.geometry.name]]  # move geometry column to last position


def h3_agg(gdf_pt: gpd.GeoDataFrame, resolution=9,
           **kwargs) -> gpd.GeoDataFrame:
    '''
    aggregates all point gdf_pt columns
    by default, calculates sum for numeric
    and unique count for non-numeric columns
    '''
    
    import h3
    import numpy as np
    
    # converting to 4326, getting coords
    gdf_pt = gdf_pt.to_crs(4326)
    # gdf_pt.geometry = gdf_pt.geometry.apply(shapely.make_valid)
    coords_tup_gen = zip(gdf_pt.geometry.y, gdf_pt.geometry.x)
    hex_gid_set = set()

    # adding new h3 indices to set
    for coords_tup in coords_tup_gen:
        hex_gid_set.add(h3.geo_to_h3(coords_tup[0], coords_tup[1], resolution=resolution))
    
    # creating function to turn h3 index to polygon, generating h3 hex gdf
    polygonise = lambda hex_id: shapely.geometry.Polygon(h3.h3_to_geo_boundary(hex_id, geo_json=True))
    gdf_hex = gpd.GeoDataFrame(pd.DataFrame({'hex_gid': list(hex_gid_set)}), # id column
                           geometry=(list(map(polygonise, list(hex_gid_set)))), # polygonized geoms
                           crs=4326)
    
    # creating (modifying) aggregation dictionary
    agg_dict = {'index_right': ['count']}
    if 'agg_dict' in kwargs: # if kwarg exists
        agg_dict.update(kwargs['agg_dict'])
        
    # we take only numeric cols to count value sums
    numeric_cols = gdf_pt.select_dtypes(include=np.number).columns.tolist()
    for col_name in numeric_cols:
        if col_name not in agg_dict:
            agg_dict[col_name] = ['sum']
    
    # dropping non-numeric columns, aggregating by hex_gid, merging to gdf_hex
    cols_to_drop = list(set(gdf_pt.columns.to_list()) - set(numeric_cols) - {gdf_pt.geometry.name})
    df_agg = gdf_hex.sjoin(gdf_pt.to_crs(4326).drop(cols_to_drop, axis=1)).groupby('hex_gid').agg(agg_dict)
    df_agg.columns = df_agg.columns.map('_'.join).str.strip('_')  # reducing multiindex to one level & renaming
    gdf_hex = gdf_hex.merge(df_agg, how='left', on='hex_gid').rename(columns={'index_right_count': 'cnt'})
    
    print(f'shape = {gdf_hex.shape} | resolution = {resolution} | crs = {gdf_hex.crs} | dropped = {cols_to_drop}')
    return gdf_hex[[col for col in gdf_hex if col not in [gdf_hex.geometry.name]] + [gdf_hex.geometry.name]]  # move geometry column to last position


def proximity_paint(gdf_base, gdf_ref, n_closest, max_dist):
    '''
    
    '''
    pass


def iso_rcost(pt_gdf: gpd.GeoDataFrame, timestep_min: list, G=False, mode='pedestrian',
              onroad_kmh_override=False, offroad_kmh_override=False, grid_res_m_override=False,
              delimiters=['water', 'military', 'industrial'], delete_holes=True,
              return_separate=True, consider_elevation=True, return_geom_type='Polygon'):
    '''
    Returns Polygon of an isochrone built by raster cost method.
    '''
    # onroad, offroad speeds, connector speed, max connector length, grid cell resolution
    speed_dict = dict(pedestrian=(5.2, 0.5, 5), bike=(22.0, 2.0, 20), drive=(60.0, 5.0, 100))
    if G:  # if we don't have a graph - get union buffered graph
        # buffer by max possible travel distance
        # load delimiters: buffer on-ground rivers
        # tobler's hiking formula - speed depending on slope
        # if return_separate=True - returns GeoDataFrame with iso geometries for each individual start point
        # if return_geometry_type='Line' - returns just the outer bounding lines for each isochrone timestep
        pass
    pass


def iso_sausage(G, pt_start: shapely.geometry.Point, ttime_minutes: list,
                travel_speed_kph=4.5, infill=True, conn_speed_kph=None,
                squash=True, edge_buff=15, node_buff=0, CRS=4326,
                network_type='walk'):
    '''
    G: expects an un-projected graph
    '''
    import osmnx as ox
    import networkx as nx
    # project graph, gather utm data
    G = ox.project_graph(G)
    UTM = G.graph['crs']

    # add startpoint as a node
    pt_start_utm = gpd.GeoDataFrame(crs=CRS, geometry=[pt_start]).to_crs(
        UTM).geometry.values[0]
    # find nearest node before start node is created - we'll need it
    # if we won't be able to split edge_nearest in two
    node_nearest = ox.distance.nearest_nodes(G, pt_start_utm.x, pt_start_utm.y)
    G.add_node('iso_pt_start', x=pt_start_utm.x, y=pt_start_utm.y,
               lon=pt_start.x, lat=pt_start.y)
    
    # find closest graph edge to pt_start_utm = edge_nearest
    edge_nearest, conn_length_m = ox.distance.nearest_edges(
        G, pt_start_utm.x, pt_start_utm.y, return_dist=True)
    if conn_length_m > 100:  # trow warning if connector length is too big
        warnings.warn(f'connector length for point {pt_start.wkt} is {round(conn_length_m)} m')

    # if nearest edge has no geometry - add it
    if 'geometry' not in G.edges[edge_nearest]:
        G.edges[edge_nearest]['geometry'] = shapely.geometry.LineString(
            [shapely.geometry.Point(x, y) for x, y in [(G.nodes[edge_nearest[0]]['x'],
                                                        G.nodes[edge_nearest[0]]['y']),
                                                       (G.nodes[edge_nearest[1]]['x'],
                                                        G.nodes[edge_nearest[1]]['y'])]])
    
    # find point on that edge closest to pt_start_utm = pt_conn_utm
    n_fr = edge_nearest[0]
    n_to = edge_nearest[1]
    pt_fr = shapely.geometry.Point(G.nodes[n_fr]['x'], G.nodes[n_fr]['y'])
    pt_to = shapely.geometry.Point(G.nodes[n_to]['x'], G.nodes[n_to]['y'])
    edge_dict = G.get_edge_data(n_fr, n_to)
    edge_lookup_near = edge_dict[min(edge_dict.keys())].get(
        "geometry", shapely.geometry.LineString([pt_fr, pt_to]))
    pt_conn_utm = shapely.ops.nearest_points(
        pt_start_utm, edge_lookup_near)[1]
    
    # make sure linestring object of nearest_edge could be split by pt_conn_utm
    G.edges[edge_nearest]['geometry'] = shapely.ops.snap(
        edge_lookup_near, pt_conn_utm, 0.0001)
    pt_conn = gpd.GeoDataFrame(crs=UTM, geometry=[pt_conn_utm]).to_crs(
        CRS).geometry.values[0]
    
    ls_collection = shapely.ops.split(G.edges[edge_nearest]['geometry'], pt_conn_utm)
    assert len(list(ls_collection.geoms)) in [1, 2], 'point has not split linestring into 1 or 2 parts'
    if conn_speed_kph:
        conn_time_local = conn_length_m / (conn_speed_kph * 1000 / 60)
    else:
        conn_time_local = conn_length_m / (travel_speed_kph * 1000 / 60)
    
    # checking if we have connector facing from us (line will not be split)
    # in that case, we draw connector edge to the nearest node to us
    if len(list(ls_collection.geoms)) == 1:  # when nearest edge is facing FROM us
        G.add_edge('iso_pt_start', node_nearest, 0, length=conn_length_m,
                   geometry=shapely.geometry.LineString([pt_start_utm, pt_conn_utm]),
                   time=conn_time_local)
    elif len(list(ls_collection.geoms)) == 2:  # classic scenario - nearest node splits edge into 2 parts
        # replace closest_edge with two edges
        
        # adding two (or four) replacement edges respecting the movement direction
        # splitting linestring of edge_nearest into ls_left and ls_right
        # add node for pt_conn
        G.add_node('iso_pt_conn', x=pt_conn_utm.x, y=pt_conn_utm.y,
                   lon=pt_conn.x, lat=pt_conn.y)
        
        # add edge from start node to pt_conn (individual speed); throw warning if length is big
        G.add_edge('iso_pt_start', 'iso_pt_conn', 0, length=conn_length_m,
                   geometry=shapely.geometry.LineString([pt_start_utm, pt_conn_utm]))
        pt_left = shapely.geometry.Point(G.nodes[edge_nearest[0]]['x'],
                                         G.nodes[edge_nearest[0]]['y'])
        pt_right = shapely.geometry.Point(G.nodes[edge_nearest[1]]['x'],
                                          G.nodes[edge_nearest[1]]['y'])
        if list(ls_collection.geoms)[0].distance(pt_left) < 1e-8:
            ls_left = list(ls_collection.geoms)[0]
            ls_right = list(ls_collection.geoms)[1]
        else:
            ls_left = list(ls_collection.geoms)[1]
            ls_right = list(ls_collection.geoms)[0]
        assert (ls_left.distance(pt_left) < 1e-8) and (
            ls_right.distance(pt_right) < 1e-8), 'error w/calc of edge_nearest geom'
        
        # either way we have to add two edges in direction of nearest_edge
        G.add_edge(edge_nearest[0], 'iso_pt_conn', 0,
                   geometry=ls_left, length=ls_left.length)
        G.add_edge('iso_pt_conn', edge_nearest[1], 0,
                   geometry=ls_right, length=ls_right.length)
        # removing one edge_nearest
        G.remove_edge(*edge_nearest)
        
        # checking if edge_nearest is bidirectional
        if G.has_edge(edge_nearest[1], edge_nearest[0], 0):
            # if bidirectional - also add the opposite direction
            G.add_edge('iso_pt_conn', edge_nearest[0], 0,
                       geometry=ls_left, length=ls_left.length)
            G.add_edge(edge_nearest[1], 'iso_pt_conn', 0,
                       geometry=ls_right, length=ls_right.length)
            # removing another edge_nearest
            G.remove_edge(edge_nearest[1], edge_nearest[0], 0)
    else:
        raise ValueError(f'length of split string is {len(list(ls_collection.geoms))}')
    
    # add speeds and travel times
    if network_type == 'walk':
        meters_per_minute = travel_speed_kph * 1000 / 60  # km per hour to m per minute
        for _, _, _, data in G.edges(data=True, keys=True):
            if 'time' not in data:
                data["time"] = data["length"] / meters_per_minute
    else:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    if conn_speed_kph and len(list(ls_collection.geoms)) == 2:  # adding proper time to connectors
        G.edges['iso_pt_start', 'iso_pt_conn', 0]['time'] = G.edges['iso_pt_start', 'iso_pt_conn', 0]['length'] / (conn_speed_kph * 1000 / 60)
    
    # get sausage isochrones
    iso_poly_dict = dict(geometry=[], iso_time=[])
    for trip_time in sorted(ttime_minutes, reverse=True):
        iso_poly_dict['iso_time'].append(trip_time)
        
        # building subgraph of where we can go within trip_time
        subgraph = nx.ego_graph(G, 'iso_pt_start', radius=trip_time, distance="time")
        
        if len(subgraph) > 1:
            # building gdf_nodes
            node_points = [shapely.geometry.Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)]
            gdf_nodes = gpd.GeoDataFrame({"id": list(subgraph.nodes)}, geometry=node_points)
            gdf_nodes = gdf_nodes.set_index("id")
    
            # building edges
            edge_lines = []
            for n_fr, n_to in subgraph.edges():
                f = gdf_nodes.loc[n_fr].geometry
                t = gdf_nodes.loc[n_to].geometry
                edge_dict = G.get_edge_data(n_fr, n_to)
                edge_lookup = edge_dict[min(edge_dict.keys())].get(
                    "geometry", shapely.geometry.LineString([f, t]))
                edge_lines.append(edge_lookup)
    
            n = gdf_nodes.buffer(node_buff).geometry
            e = gpd.GeoSeries(edge_lines).buffer(edge_buff).geometry
            all_gs = list(n) + list(e)
            new_iso = gpd.GeoSeries(all_gs).unary_union
    
            # try to fill in surrounded areas so shapes will appear solid and
            # blocks without white space inside them
            if infill:
                if new_iso.geom_type=='Polygon':
                    new_iso = shapely.geometry.MultiPolygon([shapely.geometry.Polygon(new_iso.exterior)])
                elif new_iso.geom_type=='MultiPolygon':
                    new_iso = shapely.geometry.MultiPolygon(
                        [shapely.geometry.Polygon(p.exterior) for p in list(new_iso.geoms)])
        else:  # if we have a subgraph with just a startpoint - replace it w/buffer
            new_iso = pt_start_utm.buffer(trip_time * 0.8 * travel_speed_kph * 1000 / 60)
        
        iso_poly_dict['geometry'].append(new_iso)
        
    # declare geodataframe
    gdf_iso = gpd.GeoDataFrame(iso_poly_dict, crs=UTM, geometry='geometry')
    
    # if True: squash pyramid
    if squash:
        return squash_pyramid(gdf_iso, sort_col='iso_time').to_crs(4326)
    return gdf_iso.to_crs(4326)


def iso_alpha(pt_gdf: gpd.GeoDataFrame, ttime_min: list):
    '''
    dijkstra from startpoint + concave hull - alphashape
    '''
    pass


def squash_pyramid(gdf, sort_col, *args):
    '''
    takes a gdf of overlapping polygons,
    returns gdf w/geoms that don't overlap;
    saved geoms are prioritized on sort_col
    
    first creates a sorted copy of the input;
    you can specify ascending sort mask;
    algorithm moves from top to bottom of the pyramid,
    for each row subtracting union of all previous geoms
    (subtr_geom) from current row, writing that
    donut to this row's geometry
    '''
    gdf = gdf.sort_values(by=sort_col, ascending=args[0] if args else True)
    gdf_out = gdf.copy()
    subtr_geom = gdf.loc[gdf.index[0], 'geometry']
    
    for idx, row in gdf.iloc[1:].iterrows():
        curr_geom = row['geometry']
        gdf_out.loc[idx, gdf_out.geometry.name] = curr_geom.difference(subtr_geom)
        subtr_geom = gdf_out.loc[:idx].unary_union
    return gdf_out


todo = '''
ToDo:
* drw: when giving additional params (z.b. lidewidth) renders ocean - fix
* construct_gdf: add option to drop xy cols after construction
* h3_agg: add categorical dummy feature count option w/warning if amount of cols > 10
* sjoin_lop: check if column from col_list already exists in target gdf
'''
