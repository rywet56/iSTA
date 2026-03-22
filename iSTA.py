
from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_daq as daq
from PIL import Image
import anndata as ad
import scanpy as sc
import pickle
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import time  # for hover brush throttle

# --- Button style helpers ---
BTN_ACTIVE  = {"backgroundColor": "#ff8c00", "color": "white", "border": "1px solid #d97706"}
BTN_INACTIVE = {"backgroundColor": "#e5e7eb", "color": "#111", "border": "1px solid #c5c9cf"}

# ----------------------------
def prepare_data(path):
    adata = read_py_object(path)
    cell_wall_image = adata.uns['spatial']['cell_wall_image']
    cell_segm_image = adata.uns['spatial']['staining_image_mask']
    adata.obs.reset_index(inplace=True, drop=True)
    df = adata.obs
    state_dic = adata.uns["state_dic"]
    df["track_id"] = np.arange(df.shape[0])
    if len(state_dic) == 0:
        state_dic_layer = {}
        for track_id in df["track_id"]:
            state_dic_layer[track_id] = {"size":5,
                                         "col":"blue",
                                         "visible":True,
                                         "x":df.loc[track_id,"x"],
                                         "y":df.loc[track_id,"y"],
                                         "point_id":df.loc[track_id,"point_id"],
                                         "anno_name":"",
                                         "anno_val":"",
                                         "annotated":False,
                                         "selected":False}
        state_dic["empty_anno_layer"] = state_dic_layer
        drop_down_dic = {"empty_anno_layer":["empty_annotation"]}
    else:
        drop_down_dic = adata.uns["drop_down_dic"]
    state_dic_temp = dc(state_dic["empty_anno_layer"])
    return cell_wall_image, cell_segm_image, df, adata, state_dic, drop_down_dic, state_dic_temp

def plot_data(fig, df, cell_wall_image, state_dic_layer, theta, anno_layer):
    fig = go.Figure()
    im_width = cell_wall_image.shape[1]
    im_heigth = cell_wall_image.shape[0]
    cell_wall_image = Image.fromarray(cell_wall_image)

    # add all points individually
    state_dic = state_dic_layer[anno_layer]
    for track_id in list(state_dic.keys()):
        x_coord, y_coord = df.loc[track_id, "x"], df.loc[track_id, "y"]
        fig.add_trace(
            go.Scatter(
                x=[x_coord], y=[y_coord],
                mode='markers',
                visible=True, showlegend=False,
                marker=dict(showscale=False)
            )
        )
        fig.data[track_id].marker.size = state_dic[track_id]["size"]
        fig.data[track_id].marker.color = state_dic[track_id]["col"]

    # keep zoom, disable Plotly auto-dimming
    fig.update_layout(
        template="plotly_white", width=1000, height=1000,
        xaxis_showgrid=False, yaxis_showgrid=False,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",
        hovermode="closest",
        clickmode="event"
    )
    fig.update_traces(
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=1.0))
    )

    # background image
    fig.add_layout_image(
        source=cell_wall_image,
        xref="x", yref="y",
        x=min(df["x"]), y=max(df["y"]),
        xanchor="left", yanchor="top",
        layer="below", sizing="stretch",
        sizex=im_width, sizey=im_heigth
    )
    return np.array(cell_wall_image), df, fig

def save_data(adata, df, state_dic, cell_wall_image, cell_segm_image, out_path, drop_down_dic):
    adata.obs = df
    adata.uns['spatial']['cell_wall_image'] = cell_wall_image
    adata.uns['spatial']['staining_image_mask'] = cell_segm_image
    adata.uns["state_dic"] = state_dic
    adata.uns["drop_down_dic"] = drop_down_dic
    save_py_object(adata, out_path)

def save_py_object(py_obj, path):
    with open(path, 'wb') as data_stream:
        pickle.dump(py_obj, data_stream)

def read_py_object(path):
    with open(path, 'rb') as data_stream:
        py_obj = pickle.load(data_stream)
    return py_obj

def correct_openst(path):
    adata = sc.read_h5ad(path)
    cell_wall_image = adata.uns['spatial']['cell_wall_image']
    cell_segm_image = adata.uns['spatial']['staining_image_mask']
    df = pd.DataFrame(adata.obsm['spatial'], columns=["y", "x"])
    cell_wall_image = np.array(cell_wall_image, dtype=np.uint16)
    cell_segm_image = np.array(cell_segm_image, dtype=np.uint16)
    adata.uns['spatial']['cell_wall_image'] = cell_wall_image
    adata.uns['spatial']['staining_image_mask'] = cell_segm_image
    anchor_points = np.array([
        [cell_wall_image.shape[0], 0],
        [0, 0],
        [cell_wall_image.shape[0], cell_wall_image.shape[1]],
        [0, cell_wall_image.shape[1]]
    ])
    df = np.array(df)
    df = np.concatenate((anchor_points, df))
    df = pd.DataFrame(df, columns=["y", "x"])
    df['y'] = (df['y'] - max(df['y'])) * (-1)

    adata_new = ad.concat([adata[0,:], adata], merge="same")
    adata_new = ad.concat([adata[0,:], adata_new], merge="same")
    adata_new = ad.concat([adata[0,:], adata_new], merge="same")
    adata_new.obs.reset_index(inplace=True, drop=True)
    adata_new.uns = adata.uns
    adata_new.obs["x"] = df["x"]
    adata_new.obs["y"] = df["y"]
    adata_new.uns["state_dic"] = {}
    adata_new.obs["point_id"] = adata_new.obs.index.values
    return adata_new

def highlight_point(fig, click_data, track_ids_highlighted, highlight_size, color_val, state_dic_temp):
    track_id = click_data["points"][0]["curveNumber"]
    state_dic_temp[track_id]["visible"] = True
    state_dic_temp[track_id]["selected"] = True
    state_dic_temp[track_id]["col"] = color_val
    state_dic_temp[track_id]["size"] = highlight_size
    fig.data[track_id].visible = True
    fig.data[track_id].marker.size = highlight_size
    fig.data[track_id].marker.color = color_val
    if track_id not in track_ids_highlighted:
        track_ids_highlighted.append(track_id)

def remove_highlighted_point(fig, state_dic, track_ids_highlighted, point_size, point_col, pattern_name, state_dic_temp):
    for track_id in track_ids_highlighted:
        if state_dic[pattern_name][track_id]["annotated"]:
            fig.data[track_id].marker.size = state_dic[pattern_name][track_id]["size"]
            fig.data[track_id].marker.color = state_dic[pattern_name][track_id]["col"]
        else:
            fig.data[track_id].marker.size = point_size
            fig.data[track_id].marker.color = point_col
        state_dic_temp[track_id]["selected"] = False
    track_ids_highlighted.clear()

def add_annotation(df, fig, pattern_name, pattern_value, state_dic, track_ids_highlighted,
                   drop_down_dic, state_dic_temp):
    if not pattern_name in list(drop_down_dic.keys()):
        drop_down_dic[pattern_name] = [pattern_value]
        state_dic[pattern_name] = dc(state_dic["empty_anno_layer"])
    else:
        if not pattern_value in drop_down_dic[pattern_name]:
            drop_down_dic[pattern_name].append(pattern_value)

    pattern_name_mod = "pname_" + pattern_name
    pattern_name_mod_col = "pcol_" + pattern_name
    pattern_name_mod_size = "psize_" + pattern_name

    if not pattern_name_mod in df.columns.values:
        df[pattern_name_mod] = np.full((df.shape[0], ), "")
        df[pattern_name_mod_col] = np.full((df.shape[0], ), "")
        df[pattern_name_mod_size] = np.full((df.shape[0], ), 0)

    for track_id in track_ids_highlighted:
        state_dic[pattern_name][track_id]["anno_name"] = pattern_name
        state_dic[pattern_name][track_id]["anno_val"] = pattern_value
        state_dic[pattern_name][track_id]["annotated"] = True
        state_dic[pattern_name][track_id]["col"] = fig.data[track_id].marker.color
        state_dic[pattern_name][track_id]["size"] = fig.data[track_id].marker.size
        state_dic[pattern_name][track_id]["selected"] = False

        sel = df["track_id"] == track_id
        df.loc[sel, pattern_name_mod] = pattern_value
        df.loc[sel, pattern_name_mod_col] = fig.data[track_id].marker.color
        df.loc[sel, pattern_name_mod_size] = fig.data[track_id].marker.size

        state_dic_temp[track_id]["selected"] = False
    track_ids_highlighted.clear()

def remove_annotation_val(df, fig, pattern_name, pattern_value, drop_down_dic, point_size, point_col, state_dic):
    pattern_name_mod = "pname_" + pattern_name
    pattern_name_mod_col = "pcol_" + pattern_name
    pattern_name_mod_size = "psize_" + pattern_name

    track_ids_remove = df.loc[df[pattern_name_mod] == pattern_value, "track_id"]
    for track_id in track_ids_remove:
        fig.data[track_id].marker.size = point_size
        fig.data[track_id].marker.color = point_col

        state_dic[pattern_name][track_id]["size"] = point_size
        state_dic[pattern_name][track_id]["col"] = point_col
        state_dic[pattern_name][track_id]["anno_name"] = ""
        state_dic[pattern_name][track_id]["anno_val"] = ""
        state_dic[pattern_name][track_id]["annotated"] = False

    df.loc[df[pattern_name_mod] == pattern_value, pattern_name_mod] = ""
    df.loc[df[pattern_name_mod_col] == pattern_value, pattern_name_mod_col] = ""
    df.loc[df[pattern_name_mod_size] == pattern_value, pattern_name_mod_size] = 0

    drop_down_dic[pattern_name].remove(pattern_value)

def remove_annotation_id(df, fig, pattern_name, drop_down_dic, point_size, point_col, state_dic):
    for pattern_value in drop_down_dic[pattern_name]:
        remove_annotation_val(df, fig, pattern_name, pattern_value, drop_down_dic,
                              point_size, point_col, state_dic)

    pattern_name_mod = "pname_" + pattern_name
    pattern_name_mod_col = "pcol_" + pattern_name
    pattern_name_mod_size = "psize_" + pattern_name
    df = df.drop(columns=[pattern_name_mod, pattern_name_mod_col, pattern_name_mod_size], inplace=True)
    drop_down_dic.pop(pattern_name)

def change_plot_aesthetics(fig, point_size, point_col, state_dic, current_anno_layer, state_dic_temp):
    state_dic_anno = state_dic[current_anno_layer]
    track_ids = list(state_dic_anno.keys())
    for track_id in track_ids:
        if state_dic_anno[track_id]["annotated"] or state_dic_temp[track_id]["selected"]:
            fig.data[track_id].marker.size = point_size
            state_dic_anno[track_id]["size"] = point_size
        else:
            fig.data[track_id].marker.size = point_size
            fig.data[track_id].marker.color = point_col
            state_dic_anno[track_id]["size"] = point_size
            state_dic_anno[track_id]["col"] = point_col

def change_plot_aesthetics_sel(fig, point_size, point_col, state_dic_temp):
    track_ids = list(state_dic_temp.keys())
    for track_id in track_ids:
        if state_dic_temp[track_id]["selected"]:
            fig.data[track_id].marker.size = point_size
            fig.data[track_id].marker.color = point_col
            state_dic_temp[track_id]["size"] = point_size
            state_dic_temp[track_id]["col"] = point_col

def rotate_point_cloud(df, theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array(((c, -s), (s, c)))
    coords = np.array(df.loc[:,["x","y"]])
    coords_rot = np.matmul(coords, rot_mat)
    df["x"] = coords_rot[:,0]
    df["y"] = coords_rot[:,1]
    return df

def cut_data(df, cell_wall_image, cell_segm_image, adata, x_min_new, x_max_new, y_min_new, y_max_new, state_dic, state_dic_temp):
    x_min_old, y_max_old = min(df["x"]), max(df["y"])

    idx_row_start = abs(int(np.floor(y_max_old - y_max_new)))
    idx_row_end = abs(int(np.floor(y_max_old - y_min_new)))
    idx_col_start = abs(int(np.floor(x_min_new - x_min_old)))
    idx_col_end = abs(int(np.floor(x_max_new - x_min_old)))

    up_left_ref_point = [x_min_new, y_max_new]
    bottom_left_ref_point = [x_min_new, y_min_new]
    bottom_right_ref_point = [x_max_new, y_min_new]
    up_right_ref_point = [x_max_new, y_max_new]

    adata.obs["x"] = np.array(df["x"])
    adata.obs["y"] = np.array(df["y"])

    pseudo_cells = adata[0:4,:]
    true_cells = adata[4:adata.shape[0],:]

    pseudo_cells.obs["x"] = [up_left_ref_point[0], bottom_right_ref_point[0], bottom_left_ref_point[0], up_right_ref_point[0]]
    pseudo_cells.obs["y"] = [up_left_ref_point[1], bottom_right_ref_point[1], bottom_left_ref_point[1], up_right_ref_point[1]]

    sel_x = np.logical_and(np.array(true_cells.obs["x"] >= up_left_ref_point[0]), np.array(true_cells.obs["x"] <= bottom_right_ref_point[0]))
    sel_y = np.logical_and(np.array(true_cells.obs["y"] >= bottom_right_ref_point[1]), np.array(true_cells.obs["y"] <= up_left_ref_point[1]))
    sel_cells = np.logical_and(sel_x, sel_y)

    true_cells_new = true_cells[sel_cells,:]

    adata_new = ad.concat([pseudo_cells, true_cells_new], merge="same")
    adata_new.obs.reset_index(inplace=True, drop=True)
    adata_new.uns = adata.uns

    df = adata_new.obs

    cell_wall_image = np.array(cell_wall_image)[idx_row_start:idx_row_end, idx_col_start:idx_col_end]
    cell_segm_image = np.array(cell_segm_image)[idx_row_start:idx_row_end, idx_col_start:idx_col_end]

    for anno_layer in list(state_dic.keys()):
        new_anno_layer_dic = {}
        i = 0
        for track_id in df["track_id"]:
            new_anno_layer_dic[i] = dc(state_dic[anno_layer][track_id])
            i += 1
        state_dic[anno_layer] = new_anno_layer_dic

    df["track_id"] = df.index.values

    state_dic_temp = dc(state_dic["empty_anno_layer"])
    return df, cell_wall_image, cell_segm_image, adata_new, state_dic_temp

def re_draw_graph(df, im, state_dic):
    fig = go.Figure()
    im_width = im.shape[1]
    im_heigth = im.shape[0]
    state_dic_new = {}
    im = Image.fromarray(im)

    for i in range(df.shape[0]):
        x_coord, y_coord = df.loc[i, "x"], df.loc[i, "y"]
        fig.add_trace(
            go.Scatter(
                x=[x_coord], y=[y_coord],
                mode='markers',
                visible=True, showlegend=False,
                marker=dict(size=4, color="blue", showscale=False)
            )
        )
        state_dic_new[i] = {"size":5, "col":"blue", "visible":True, "anno_name":"", "anno_val":"", "annotated":False, "selected":False}

    fig.update_layout(
        template="plotly_white", width=1000, height=1000,
        xaxis_showgrid=False, yaxis_showgrid=False,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",
        hovermode="closest",
        clickmode="event"
    )
    fig.update_traces(
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=1.0))
    )

    fig.add_layout_image(
        source=im,
        xref="x", yref="y",
        x=min(df["x"]), y=max(df["y"]),
        xanchor="left", yanchor="top",
        layer="below", sizing="stretch",
        sizex=im_width, sizey=im_heigth
    )
    return fig, state_dic_new

def rotate_data(cell_wall_image, cell_segm_image, df, theta):
    cell_wall_image = Image.fromarray(cell_wall_image)
    cell_segm_image = Image.fromarray(cell_segm_image)
    cell_wall_image = cell_wall_image.rotate(theta, expand=True)
    cell_segm_image = cell_segm_image.rotate(theta, expand=True)
    df = rotate_point_cloud(df, -theta)
    return np.array(cell_wall_image), np.array(cell_segm_image), df

# ---------- Brush & Lasso helpers ----------
def highlight_point_with_neighbors(fig, hover_or_click_data, track_ids_highlighted,
                                   highlight_size, color_val, state_dic_temp,
                                   df, brush_radius_px: float = 0.0):
    """Single-point highlight + optional neighbor radius around it."""
    if not hover_or_click_data or "points" not in hover_or_click_data:
        return
    # center point
    highlight_point(fig, hover_or_click_data, track_ids_highlighted, highlight_size, color_val, state_dic_temp)

    # neighbors (optional)
    if brush_radius_px and brush_radius_px > 0 and len(df) > 0:
        track_id = hover_or_click_data["points"][0]["curveNumber"]
        X = df["x"].to_numpy(float); Y = df["y"].to_numpy(float)
        cx, cy = float(X[track_id]), float(Y[track_id])
        R2 = float(brush_radius_px)**2
        d2 = (X - cx)**2 + (Y - cy)**2
        neigh_ids = np.where(d2 <= R2)[0].tolist()
        for nid in neigh_ids:
            fake_click = {"points": [{"curveNumber": int(nid)}]}
            highlight_point(fig, fake_click, track_ids_highlighted, highlight_size, color_val, state_dic_temp)

def highlight_points_from_selectedData(fig, selectedData, track_ids_highlighted,
                                       highlight_size, color_val, state_dic_temp,
                                       df, brush_radius_px: float = 0.0):
    """
    Highlight points from lasso/box 'selectedData'.
    Optionally expand selection by neighbor radius (in data units).
    """
    if not selectedData or "points" not in selectedData:
        return
    centers = []
    for p in selectedData["points"]:
        centers.append(p["curveNumber"])
        fake_click = {"points": [{"curveNumber": p["curveNumber"]}]}
        highlight_point(fig, fake_click, track_ids_highlighted, highlight_size, color_val, state_dic_temp)

    if brush_radius_px and brush_radius_px > 0 and len(df) > 0:
        X = df["x"].to_numpy(float); Y = df["y"].to_numpy(float)
        R2 = float(brush_radius_px)**2
        for c in centers:
            cx, cy = float(X[c]), float(Y[c])
            d2 = (X - cx)**2 + (Y - cy)**2
            neigh_ids = np.where(d2 <= R2)[0].tolist()
            for nid in neigh_ids:
                fake_click = {"points": [{"curveNumber": int(nid)}]}
                highlight_point(fig, fake_click, track_ids_highlighted, highlight_size, color_val, state_dic_temp)

# define external .css style sheet to make buttons and fields look better
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# create the app instance
app = Dash(__name__, external_stylesheets=external_stylesheets)
track_ids_highlighted = []
selection_mode = "click"  # "click" or "brush" or "lasso"
HOVER_THROTTLE_SEC = 0.06  # ~16 Hz
last_hover_ts = 0.0

input_path = '/Users/manuelneumann/Library/CloudStorage/SeaDrive-ManuelNeumann(box.hu-berlin.de)/My Libraries/spatial_expression/For_Paper/Real_Data/Completed_Section_Annotations/R4_S1/R4_S1_stitched_segmented_corrected_anno.pkl'
out_put_path = '/Users/manuelneumann/Library/CloudStorage/SeaDrive-ManuelNeumann(box.hu-berlin.de)/My Libraries/spatial_expression/For_Paper/Real_Data/Completed_Section_Annotations/R4_S1/R4_S1_stitched_segmented_corrected_anno.pkl'

# input_path = '/Users/manuelneumann/Library/CloudStorage/SeaDrive-ManuelNeumann(box.hu-berlin.de)/My Libraries/spatial_expression/For_Paper/Real_Data/Completed_Section_Annotations/R5_S2/R5_S2_stitched_segmented_corrected_anno.pkl'
# out_put_path = '/Users/manuelneumann/Library/CloudStorage/SeaDrive-ManuelNeumann(box.hu-berlin.de)/My Libraries/spatial_expression/For_Paper/Real_Data/Completed_Section_Annotations/R5_S2/R5_S2_stitched_segmented_corrected_anno.pkl'

current_anno_id = "empty_anno_layer"
current_anno_val = "empty_annotation"
drop_down_dic = {"empty_anno_layer":["empty_annotation"]}
rotation_optins = [0,90,180,270]
fig = go.Figure()

# ------------------ LAYOUT ------------------
app.layout = html.Div([
    html.Div([
        html.Div(["Read in Point Cloud data"], style={"font-size":"20px", "font-weight":"bold"}),
        html.Div(["Input Path: ", dcc.Input(id='input_path', value=input_path, type='text', size="100")]),
        html.Button(id='import_data_button', n_clicks=0, children="Import Data"),
        html.Button(id='plot_data_button', n_clicks=0, children="Plot Data"),
    ]),
    html.Div([
        html.Div([
            html.Div(["Rotation"]),
            html.Div([
                html.Div([dcc.Dropdown(rotation_optins, id='drop-down_rotation')], style={'width': '10%'}),
                html.Div([html.Button(id='rotate', n_clicks=0, children="Rotate Data")]),
            ], style={'display': 'flex'}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Div(["x min: ", dcc.Input(id='x_min', value=-1000, type='number')]),
                        html.Div(["x max: ", dcc.Input(id='x_max', value=-1000, type='number')]),
                    ], style={'display': 'flex'}),
                    html.Div([
                        html.Div(["y min: ", dcc.Input(id='y_min', value=1500, type='number')]),
                        html.Div(["y max: ", dcc.Input(id='y_max', value=2000, type='number')]),
                    ], style={'display': 'flex'}),
                ]),
                html.Div([html.Button(id='cut_data_button', n_clicks=0, children="Cut Data")],
                         style={'justifyContent': 'center', "align-items": "center", 'display': 'flex'}),
            ], style={'display': 'flex'}),

            # --- Selection controls ---
            # Row 1: Toggle + 3 buttons (same line)
            html.Div([
                daq.ToggleSwitch(
                    id='click_mode_button',
                    label="Highlight Mode ",
                    value=False,
                    color="red"
                ),
                html.Button(id='select_click_mode', n_clicks=0, children="Click Select", style=BTN_INACTIVE),
                html.Button(id='select_brush_mode', n_clicks=0, children="Brush Select", style=BTN_INACTIVE),
                html.Button(id='select_lasso_mode', n_clicks=0, children="Lasso Select", style=BTN_INACTIVE),
            ], style={"alignItems": "center", 'display': 'flex', "gap": "8px", "paddingLeft": "12px"}),

            # Row 2: Brush radius below
            html.Div([
                html.Div(["Brush radius (px): ",
                          dcc.Input(id='brush_radius_px', value=0, type='number', min=0)],
                         style={'marginTop': '10px'})
            ], style={'paddingLeft': '12px'}),

            # Row 3: Remove highlight below
            html.Div([
                html.Button(id='remove_highlight', n_clicks=0, children="Remove Highlight")
            ], style={"alignItems": "center", 'display': 'flex', "paddingLeft": "12px", "marginTop": "10px"}),

            # General Aesthetics
            html.Div([
                html.Div([
                    html.Div(["point size: "]),
                    html.Div([dcc.Input(id='point_size', value=5, type='number')]),
                ]),
                html.Div([
                    html.Div(["Point Color: "]),
                    html.Div([dcc.Input(id='point_col', value="blue", type='text')]),
                ]),
                html.Div([html.Button(id='change_aesthetics', n_clicks=0, children="Change Aeshetic")],
                         style={"align-items": "flex-end", 'display': 'flex'}),
            ], style={'display': 'flex'}),

            # Selection Aesthetics
            html.Div([
                html.Div([
                    html.Div(["Point Size Select: "]),
                    html.Div([dcc.Input(id='point_size_select', value=5, type='number')]),
                ]),
                html.Div([
                    html.Div(["Point Color Select: "]),
                    html.Div([dcc.Input(id='point_col_select', value="red", type='text')]),
                ]),
                html.Div([html.Button(id='change_aesthetics_sel', n_clicks=0, children="Change Highlight Aeshetics")],
                         style={"align-items": "flex-end", 'display': 'flex'}),
            ], style={'display': 'flex'}),

            # Adding Annotation
            html.Div([
                html.Div(["Annotatoin Name: ", dcc.Input(id='pattern_name', value="organs", type='text')]),
                html.Div(["ID Input: ", dcc.Input(id='pattern_val', value="sepal", type='text')]),
                html.Div([html.Button(id='add_anno', n_clicks=0, children="Add Annotation")],
                         style={"align-items": "flex-end", 'display': 'flex'}),
            ], style={'display': 'flex'}),

            # Annotation Drop-Down Menus
            html.Div([
                html.Div([
                    html.Div(["Annotation Track", dcc.Dropdown(list(drop_down_dic.keys()), current_anno_id, id='drop-down_anno_id')], style={'width': '100%'}),
                    dcc.Interval(id='interval_component_anno_id', interval=1*1000, n_intervals=0),
                    html.Div([html.Button(id='remove_anno_id', n_clicks=0, children="Remove Annotation Track")]),
                ]),
                html.Div([
                    html.Div(["Annotation Value", dcc.Dropdown(drop_down_dic["empty_anno_layer"], current_anno_val, id='drop-down_anno_val')], style={'width': '100%'}),
                    dcc.Interval(id='interval_component_anno_val', interval=1*1000, n_intervals=0),
                    html.Div([html.Button(id='remove_anno_val', n_clicks=0, children="Remove Annotation Value")]),
                ], style={'padding-left': 10})
            ], style={'display': 'flex'}),
        ]),

        # place for plot
        html.Div([
            dcc.Graph(
                id="graph",
                config={
                    "doubleClick": "reset",
                    "displaylogo": False,
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "modeBarButtonsToAdd": ["lasso2d", "select2d"]
                }
            )
        ], style={'justifyContent': 'center','display': 'flex'}),
    ], style={'display': 'flex'}),

    # Save Data
    html.Div([
        html.Br(),
        html.Div(["Save Results"], style={"font-size":"20px", "font-weight":"bold"}),
        html.Div(["Output Path: ", dcc.Input(id='output_path', value=out_put_path, type='text', size="100")]),
        html.Button(id='save_data_button', n_clicks=0, children="Save Data"),
    ])
])

@app.callback(
    Output(component_id='drop-down_anno_val', component_property='options'),
    Input(component_id='interval_component_anno_val', component_property='n_intervals'),
    State(component_id='drop-down_anno_id', component_property='value'),
    prevent_initial_call=True
)
def update_drop_down_anno_val(update_dropdown, anno_id):
    if anno_id in drop_down_dic.keys():
        return drop_down_dic[anno_id]
    else:
        return []

@app.callback(
    Output(component_id='drop-down_anno_id', component_property='options'),
    Input(component_id='interval_component_anno_id', component_property='n_intervals'),
    prevent_initial_call=True
)
def update_drop_down_anno_id(update_dropdown):
    return list(drop_down_dic.keys())

@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id='import_data_button', component_property='n_clicks'),
    State(component_id='input_path', component_property='value'),
    Input(component_id='plot_data_button', component_property='n_clicks'),
    Input(component_id='graph', component_property='clickData'),
    Input(component_id='graph', component_property='selectedData'),
    Input(component_id='graph', component_property='hoverData'),
    State(component_id='graph', component_property='relayoutData'),
    State(component_id='click_mode_button', component_property='value'),
    State(component_id='point_size', component_property='value'),
    State(component_id='point_col', component_property='value'),
    Input(component_id='change_aesthetics', component_property='n_clicks'),
    Input(component_id='change_aesthetics_sel', component_property='n_clicks'),
    State(component_id='point_size_select', component_property='value'),
    State(component_id='point_col_select', component_property='value'),
    State(component_id='pattern_name', component_property='value'),
    State(component_id='pattern_val', component_property='value'),
    Input(component_id='add_anno', component_property='n_clicks'),
    Input(component_id='remove_anno_id', component_property='n_clicks'),
    Input(component_id='remove_anno_val', component_property='n_clicks'),
    Input(component_id='remove_highlight', component_property='n_clicks'),
    Input(component_id='drop-down_anno_id', component_property='value'),
    State(component_id='drop-down_anno_val', component_property='value'),
    Input(component_id='cut_data_button', component_property='n_clicks'),
    State(component_id='x_min', component_property='value'),
    State(component_id='x_max', component_property='value'),
    State(component_id='y_min', component_property='value'),
    State(component_id='y_max', component_property='value'),
    Input(component_id='rotate', component_property='n_clicks'),
    State(component_id='drop-down_rotation', component_property='value'),
    Input(component_id='save_data_button', component_property='n_clicks'),
    State(component_id='output_path', component_property='value'),
    Input(component_id='select_click_mode', component_property='n_clicks'),
    Input(component_id='select_brush_mode', component_property='n_clicks'),
    Input(component_id='select_lasso_mode', component_property='n_clicks'),
    State(component_id='brush_radius_px', component_property='value'),
    prevent_initial_call=True
)
def update_plot(import_data_signal, input_path_val, plot_data_signal,
                click_data, selected_data, hover_data, relayout_data,
                highlight_mode,
                point_size, point_col,
                change_aesthetics_signal,
                change_aesthetics_sel_signal, point_size_select, point_col_select,
                pattern_name, pattern_val,
                add_anno, rem_anno_id, rem_anno_val,
                rem_highlight,
                drop_down_menu_anno_id, drop_down_menu_anno_val,
                redraw_signal, xmin, xmax, ymin, ymax,
                rotate_signal, rotation_angle,
                save_data_signal, out_path,
                select_click_signal, select_brush_signal, select_lasso_signal,
                brush_radius_px
                ):
    triggered_id = ctx.triggered_id

    global df
    global cell_wall_image
    global cell_segm_image
    global adata
    global fig
    global state_dic
    global track_ids_highlighted
    global drop_down_dic
    global state_dic_temp
    global selection_mode
    global last_hover_ts

    if triggered_id == "import_data_button":
        cell_wall_image, cell_segm_image, df, adata, state_dic, drop_down_dic, state_dic_temp = prepare_data(input_path_val)

    elif triggered_id == "plot_data_button":
        cell_wall_image, df, fig = plot_data(fig, df, cell_wall_image, state_dic, 0, anno_layer=drop_down_menu_anno_id)
        fig.update_layout(dragmode=('lasso' if selection_mode == "lasso" else 'zoom'))

    elif triggered_id == "select_click_mode":
        selection_mode = "click"
        if fig: fig.update_layout(dragmode='zoom')

    elif triggered_id == "select_brush_mode":
        selection_mode = "brush"
        if fig: fig.update_layout(dragmode='zoom')

    elif triggered_id == "select_lasso_mode":
        selection_mode = "lasso"
        if fig: fig.update_layout(dragmode='lasso')

    elif highlight_mode and triggered_id == "graph":
        if selection_mode == "brush" and hover_data:
            now = time.time()
            if (now - last_hover_ts) >= HOVER_THROTTLE_SEC:
                last_hover_ts = now
                highlight_point_with_neighbors(
                    fig, hover_data, track_ids_highlighted,
                    point_size_select, point_col_select, state_dic_temp,
                    df, brush_radius_px or 0.0
                )
        elif selection_mode == "lasso" and selected_data:
            highlight_points_from_selectedData(
                fig, selected_data, track_ids_highlighted,
                point_size_select, point_col_select, state_dic_temp,
                df, brush_radius_px or 0.0
            )
        elif selection_mode == "click" and click_data:
            highlight_point(fig, click_data, track_ids_highlighted,
                            point_size_select, point_col_select, state_dic_temp)

    elif triggered_id == "add_anno":
        add_annotation(df, fig, pattern_name, pattern_val,
                       state_dic, track_ids_highlighted,
                       drop_down_dic, state_dic_temp)

    elif triggered_id == "remove_anno_id":
        remove_annotation_id(df, fig, drop_down_menu_anno_id, drop_down_dic,
                             point_size, point_col, state_dic)

    elif triggered_id == "remove_anno_val":
        remove_annotation_val(df, fig, drop_down_menu_anno_id, drop_down_menu_anno_val, drop_down_dic,
                              point_size, point_col, state_dic)

    elif triggered_id == "remove_highlight":
        remove_highlighted_point(fig, state_dic, track_ids_highlighted, point_size, point_col, drop_down_menu_anno_id, state_dic_temp)

    elif triggered_id == "change_aesthetics":
        change_plot_aesthetics(fig, point_size, point_col, state_dic, drop_down_menu_anno_id, state_dic_temp)

    elif triggered_id == "change_aesthetics_sel":
        change_plot_aesthetics_sel(fig, point_size_select, point_col_select, state_dic_temp)

    elif triggered_id == "cut_data_button":
        df, cell_wall_image, cell_segm_image, adata, state_dic_temp = cut_data(df, cell_wall_image, cell_segm_image, adata, xmin, xmax, ymin, ymax, state_dic, state_dic_temp)
        cell_wall_image, df, fig = plot_data(fig, df, cell_wall_image, state_dic, 0, anno_layer=drop_down_menu_anno_id)
        fig.update_layout(dragmode=('lasso' if selection_mode == "lasso" else 'zoom'))

    elif triggered_id == "rotate":
        cell_wall_image, cell_segm_image, df = rotate_data(cell_wall_image, cell_segm_image, df, theta=rotation_angle)
        cell_wall_image, df, fig = plot_data(fig, df, cell_wall_image, state_dic, 0, anno_layer=drop_down_menu_anno_id)
        fig.update_layout(dragmode=('lasso' if selection_mode == "lasso" else 'zoom'))

    elif triggered_id == "save_data_button":
        save_data(adata, df, state_dic, cell_wall_image, cell_segm_image, out_path, drop_down_dic)

    elif triggered_id == "drop-down_anno_id":
        cell_wall_image, df, fig = plot_data(fig, df, cell_wall_image, state_dic, 0, anno_layer=drop_down_menu_anno_id)
        fig.update_layout(dragmode=('lasso' if selection_mode == "lasso" else 'zoom'))

    # Preserve current zoom/pan
    if isinstance(relayout_data, dict) and fig:
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            fig['layout']['xaxis']['range'] = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
        if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
            fig['layout']['yaxis']['range'] = [relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]

    return fig

# --- UI feedback for Click/Brush/Lasso buttons ---
@app.callback(
    Output('select_click_mode', 'style'),
    Output('select_brush_mode', 'style'),
    Output('select_lasso_mode', 'style'),
    Input('select_click_mode', 'n_clicks'),
    Input('select_brush_mode', 'n_clicks'),
    Input('select_lasso_mode', 'n_clicks'),
    Input('click_mode_button', 'value'),  # Highlight Mode toggle
    prevent_initial_call=False
)
def selection_mode_button_styles(nc_click, nc_brush, nc_lasso, highlight_on):
    """
    Visual feedback:
    - If Highlight Mode is OFF: all three buttons look inactive (grey).
    - If ON: the most recently pressed button is active (orange); the others are grey.
    """
    if not highlight_on:
        return BTN_INACTIVE, BTN_INACTIVE, BTN_INACTIVE

    nc_click = nc_click or 0
    nc_brush = nc_brush or 0
    nc_lasso = nc_lasso or 0

    # Find max n_clicks; tie-breaker priority: click > brush > lasso
    max_n = max(nc_click, nc_brush, nc_lasso)
    click_on = (nc_click == max_n and max_n > 0)
    brush_on = (nc_brush == max_n and max_n > 0 and not click_on)
    lasso_on = (nc_lasso == max_n and max_n > 0 and not click_on and not brush_on)

    return (
        BTN_ACTIVE if click_on else BTN_INACTIVE,
        BTN_ACTIVE if brush_on else BTN_INACTIVE,
        BTN_ACTIVE if lasso_on else BTN_INACTIVE
    )

if __name__ == '__main__':
    app.run(debug=True, port=8051)
