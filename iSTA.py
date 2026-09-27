
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL
from dash.exceptions import PreventUpdate
import re
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_daq as daq
from PIL import Image
import anndata as ad
import scanpy as sc
import pickle
import os
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import time  # for hover brush throttle

# --- Button style helpers ---
BTN_ACTIVE  = {"backgroundColor": "#ff8c00", "color": "white", "border": "1px solid #d97706"}
BTN_INACTIVE = {"backgroundColor": "#e5e7eb", "color": "#111", "border": "1px solid #c5c9cf"}
IMPORT_IDLE = {"backgroundColor": "#2563eb", "color": "white", "border": "1px solid #1d4ed8"}
IMPORT_BUSY = {"backgroundColor": "#1e3a8a", "color": "white", "border": "1px solid #1e3a8a"}
PLOT_IDLE = {"backgroundColor": "#f97316", "color": "white", "border": "1px solid #ea580c"}
PLOT_BUSY = {"backgroundColor": "#c2410c", "color": "white", "border": "1px solid #9a3412"}
IO_STATUS_ON = {"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "8px", "fontSize": "14px"}
IO_STATUS_OFF = {"display": "none"}
TAB_ACTIVE = {
    "backgroundColor": "#ff8c00", "color": "white", "border": "1px solid #d97706",
    "borderRadius": "6px 6px 0 0", "padding": "10px 16px", "marginRight": "6px",
}
TAB_INACTIVE = {
    "backgroundColor": "#e5e7eb", "color": "#111", "border": "1px solid #c5c9cf",
    "borderRadius": "6px 6px 0 0", "padding": "10px 16px", "marginRight": "6px",
}
PANEL_SHOW = {"display": "block"}
PANEL_HIDE = {"display": "none"}
MENU_OPEN = {"display": "block"}
MENU_CLOSED = {"display": "none"}
TYPE_OPEN = {"display": "block"}
TYPE_CLOSED = {"display": "none"}

# Display labels are what the dropdown shows. The key is written into columns as pname_<key>.
ANNO_LAYERS = {
    "flower_id": {
        "label": "Flower ID",
        "values": [
            "blank", "f_0.5", "f_1", "F_1", "F_1_5", "f_10", "f_11", "f_12", "f_14",
            "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "IM", "stem",
        ],
    },
    "organ_domain_id": {
        "label": "Organ domain",
        "values": ["IM", "inner_layer", "outer_layer", "pith", "sepal", "stamen_carpel"],
    },
    "layer_id": {
        "label": "Layer",
        "values": ["blank", "bgr", "L1", "L2", "L3"],
    },
    "cell_type_layer": {
        "label": "Cell type",
        "values": [
            "blank", "central_pedicel_cell", "FM1_stem_cell", "FM2_stem_cell",
            "IM_stem_cell", "pith_cell",
        ],
    },
}
BUILTIN_LAYERS = set(ANNO_LAYERS)
BUILTIN_VALUES = {key: set(spec["values"]) for key, spec in ANNO_LAYERS.items()}

def layer_dropdown_options():
    return [
        {"label": spec["label"], "value": key, "removable": key not in BUILTIN_LAYERS}
        for key, spec in ANNO_LAYERS.items()
    ]

def value_dropdown_options(layer_key):
    spec = ANNO_LAYERS.get(layer_key) or {"values": []}
    builtin = BUILTIN_VALUES.get(layer_key, set())
    return [
        {"label": value, "value": value, "removable": value not in builtin}
        for value in spec["values"]
    ]

def ensure_annotation_layer(label):
    text = (label or "").strip()
    for key, spec in ANNO_LAYERS.items():
        if key == text or spec["label"].lower() == text.lower():
            return key
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_") or "layer"
    base = slug
    n = 2
    while slug in ANNO_LAYERS and ANNO_LAYERS[slug]["label"] != text:
        slug = f"{base}_{n}"
        n += 1
    if slug not in ANNO_LAYERS:
        ANNO_LAYERS[slug] = {"label": text, "values": []}
    return slug

def option_rows(kind, options, selected):
    rows = []
    for opt in options:
        selected_cls = " ista-opt-selected" if opt["value"] == selected else ""
        row = [
            html.Button(
                opt["label"],
                id={"type": f"{kind}-pick", "index": opt["value"]},
                n_clicks=0,
                className="ista-opt-label",
            ),
        ]
        if opt.get("removable"):
            row.append(html.Button(
                "×",
                id={"type": f"{kind}-remove", "index": opt["value"]},
                n_clicks=0,
                className="ista-opt-x",
                title="Remove",
            ))
        rows.append(html.Div(row, className="ista-opt" + selected_cls))
    return rows

def creatable_picker(title, value_id, kind, options, value):
    label = next((opt["label"] for opt in options if opt["value"] == value), value or "Select")
    return html.Div([
        html.Div(title, className="ista-drop-caption"),
        dcc.Input(id=value_id, value=value, type="text", style={"display": "none"}),
        html.Div([
            html.Div([
                html.Button(label, id=f"{value_id}_display", n_clicks=0, className="ista-drop-display"),
                dcc.Input(
                    id=f"{value_id}_type",
                    type="text",
                    placeholder="Type, then Enter",
                    n_submit=0,
                    className="ista-drop-type",
                    style=TYPE_CLOSED,
                ),
            ], className="ista-drop-field"),
            html.Button(
                html.Span("+", className="ista-plus-glyph"),
                id=f"{value_id}_plus",
                n_clicks=0,
                className="ista-plus",
                title="Add",
            ),
        ], className="ista-drop"),
        html.Div(
            option_rows(kind, options, value),
            id=f"{value_id}_menu",
            className="ista-drop-menu",
            style=MENU_CLOSED,
        ),
        dcc.Store(id=f"{value_id}_focus"),
    ], className="ista-picker")

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
        template="plotly_white", autosize=True,
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

def list_data_files(directory):
    """Pickle files in the working directory, including one level of subfolders."""
    if not directory or not os.path.isdir(directory):
        return []
    found = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return []
    for name in entries:
        if name.startswith("."):
            continue
        full = os.path.join(directory, name)
        if os.path.isfile(full) and name.endswith(".pkl"):
            found.append(name)
        elif os.path.isdir(full):
            try:
                children = sorted(os.listdir(full))
            except OSError:
                continue
            for child in children:
                if child.startswith(".") or not child.endswith(".pkl"):
                    continue
                child_path = os.path.join(full, child)
                if os.path.isfile(child_path):
                    found.append(os.path.join(name, child))
    return found

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
        template="plotly_white", autosize=True,
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
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .ista-picker { margin: 0 16px 8px 0; min-width: 240px; flex: 1; max-width: 420px; position: relative; }
            .ista-drop-caption { font-size: 13px; margin-bottom: 4px; color: #374151; }
            .ista-drop {
                display: flex; align-items: stretch; height: 36px;
                border: 1px solid #d1d5db; border-radius: 10px; background: #fff; overflow: hidden;
            }
            .ista-drop-field { flex: 1; min-width: 0; }
            .ista-drop-display, .ista-drop-type {
                width: 100% !important; height: 36px !important; margin: 0 !important;
                padding: 0 28px 0 12px !important; border: none !important; border-radius: 0 !important;
                background: transparent !important; box-shadow: none !important;
                text-align: left !important; text-transform: none !important; letter-spacing: normal !important;
                font-size: 14px !important; font-weight: 500 !important; line-height: 36px !important;
                color: #111827 !important;
            }
            .ista-drop-display { position: relative; }
            .ista-drop-display::after {
                content: "▾"; position: absolute; right: 10px; top: 0;
                color: #6b7280; font-size: 12px;
            }
            .ista-plus {
                width: 36px !important; height: 36px !important; min-width: 36px !important; max-width: 36px !important;
                margin: 0 !important; padding: 0 !important; border: none !important;
                border-left: 1px solid #e5e7eb !important; border-radius: 0 !important;
                background: #111827 !important; color: #fff !important;
                display: flex !important; align-items: center !important; justify-content: center !important;
                box-sizing: border-box !important; line-height: 1 !important;
                letter-spacing: 0 !important; text-transform: none !important;
            }
            .ista-plus-glyph {
                display: block; font-size: 22px; font-weight: 400; line-height: 1;
                width: 1em; text-align: center;
            }
            .ista-drop-menu {
                position: absolute; z-index: 20; left: 0; right: 16px;
                margin-top: 4px; max-height: 240px; overflow: auto;
                background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
                box-shadow: 0 8px 24px rgba(0,0,0,.08);
            }
            .ista-opt { display: flex; align-items: center; gap: 4px; padding: 2px 4px; }
            .ista-opt-selected { background: #fff7ed; }
            .ista-opt-label {
                flex: 1; height: 32px !important; margin: 0 !important; padding: 0 8px !important;
                border: none !important; border-radius: 6px !important; background: transparent !important;
                text-align: left !important; text-transform: none !important; letter-spacing: normal !important;
                font-size: 14px !important; font-weight: 500 !important; line-height: 32px !important;
                color: #111827 !important; box-shadow: none !important;
            }
            .ista-opt-x {
                width: 28px !important; height: 28px !important; min-width: 28px !important; max-width: 28px !important;
                margin: 0 !important; padding: 0 !important; border: none !important; border-radius: 6px !important;
                background: transparent !important; color: #6b7280 !important;
                display: flex !important; align-items: center !important; justify-content: center !important;
                font-size: 18px !important; line-height: 1 !important; letter-spacing: 0 !important;
                text-transform: none !important; box-shadow: none !important;
            }
            .ista-opt-x:hover { background: #fee2e2 !important; color: #b91c1c !important; }
            .ista-inline-btn {
                margin: 0 !important;
                flex: 0 0 auto;
                white-space: nowrap;
                height: 38px !important;
                padding: 0 14px !important;
                letter-spacing: .04rem !important;
                cursor: pointer;
            }
            button.ista-btn-import:active { background-color: #1e3a8a !important; }
            button.ista-btn-plot:active { background-color: #c2410c !important; }
            button.ista-btn-import:disabled,
            button.ista-btn-plot:disabled { opacity: 1 !important; cursor: wait !important; }
            .ista-spin {
                width: 16px; height: 16px; flex: 0 0 auto;
                border: 2px solid rgba(128,128,128,.35);
                border-top-color: #2563eb;
                border-radius: 50%;
                animation: ista-spin .7s linear infinite;
            }
            @keyframes ista-spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
track_ids_highlighted = []
selection_mode = "click"  # "click" or "brush" or "lasso"
HOVER_THROTTLE_SEC = 0.06  # ~16 Hz
last_hover_ts = 0.0

working_directory = '/Users/manuelneumann/Library/CloudStorage/SeaDrive-ManuelNeumann(box.hu-berlin.de)/My Libraries/spatial_expression/old/For_Paper/Real_Data/Completed_Section_Annotations'
default_input_file = 'R3_S2/R3_S2_stitched_segmented_corrected_anno.pkl'
input_file_options = list_data_files(working_directory)
if default_input_file not in input_file_options and input_file_options:
    default_input_file = input_file_options[0]
default_output_file = default_input_file if default_input_file in input_file_options else None

current_anno_id = "empty_anno_layer"
current_anno_val = "empty_annotation"
drop_down_dic = {"empty_anno_layer":["empty_annotation"]}
rotation_optins = [0,90,180,270]
fig = go.Figure()
fig.update_layout(template="plotly_white", autosize=True, margin=dict(l=0, r=0, t=0, b=0))

# ------------------ LAYOUT ------------------
app.layout = html.Div([
    html.Div([
        html.Button("Read-write data", id="tab_read_write", n_clicks=0, style=TAB_ACTIVE),
        html.Button("Rotation and cutting", id="tab_rotation_cutting", n_clicks=0, style=TAB_INACTIVE),
        html.Button("Annotation", id="tab_annotation", n_clicks=0, style=TAB_INACTIVE),
    ], style={"display": "flex", "flexDirection": "row", "alignItems": "flex-end", "marginBottom": "8px"}),
    html.Div([
        html.Div([
            html.Div([
                html.Div(["Read in Point Cloud data"], style={"font-size":"20px", "font-weight":"bold"}),
                html.Div([
                    html.Div("Working directory:"),
                    dcc.Input(
                        id='working_directory',
                        value=working_directory,
                        type='text',
                        style={"width": "100%", "maxWidth": "none", "boxSizing": "border-box"},
                    ),
                ], style={"marginTop": "6px", "marginBottom": "6px"}),
                html.Div([
                    html.Div(
                        dcc.Dropdown(
                            id='input_file',
                            options=input_file_options,
                            value=default_input_file if default_input_file in input_file_options else None,
                            clearable=False,
                            placeholder="Choose a file",
                        ),
                        style={"flex": "1", "minWidth": "0"},
                    ),
                    html.Button(
                        id='import_data_button', n_clicks=0, children="Import Data",
                        className="ista-inline-btn ista-btn-import", style=IMPORT_IDLE,
                    ),
                    html.Button(
                        id='plot_data_button', n_clicks=0, children="Plot Data",
                        className="ista-inline-btn ista-btn-plot", style=PLOT_IDLE,
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "8px", "marginBottom": "4px"}),
                html.Div(
                    [html.Span(className="ista-spin"), html.Span(id="io_status_text")],
                    id="io_status",
                    style=IO_STATUS_OFF,
                ),
                html.Div(["Save Results"], style={"font-size":"20px", "font-weight":"bold", "marginTop": "16px"}),
                html.Div([
                    html.Div(
                        dcc.Dropdown(
                            id='output_file',
                            options=input_file_options,
                            value=default_output_file,
                            clearable=False,
                            searchable=True,
                            placeholder="Save as",
                        ),
                        style={"flex": "1", "minWidth": "0"},
                    ),
                    html.Button(id='save_data_button', n_clicks=0, children="Save Data", className="ista-inline-btn"),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "8px"}),
            ], id="panel_read_write", style=PANEL_SHOW),
            html.Div([
                html.Div(["Rotation"]),
                html.Div([
                    html.Div([dcc.Dropdown(rotation_optins, id='drop-down_rotation')], style={'width': '140px'}),
                    html.Div([html.Button(id='rotate', n_clicks=0, children="Rotate Data")]),
                ], style={'display': 'flex', "gap": "8px", "alignItems": "center"}),

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
            ], id="panel_rotation_cutting", style=PANEL_HIDE),

            html.Div([
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

                html.Div([
                    html.Div(["Brush radius (px): ",
                              dcc.Input(id='brush_radius_px', value=0, type='number', min=0)]),
                    html.Button(id='remove_highlight', n_clicks=0, children="Remove Highlight"),
                ], style={"alignItems": "center", "display": "flex", "gap": "8px", "paddingLeft": "12px", "marginTop": "10px"}),

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

                html.Div([
                    creatable_picker("Annotation", "pattern_name", "layer", layer_dropdown_options(), "flower_id"),
                    creatable_picker("ID", "pattern_val", "val", value_dropdown_options("flower_id"), "blank"),
                    html.Div(
                        [html.Button(id='add_anno', n_clicks=0, children="Add Annotation")],
                        style={"alignItems": "flex-end", "display": "flex", "marginBottom": "8px"},
                    ),
                ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end", "marginTop": "12px"}),

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
            ], id="panel_annotation", style=PANEL_HIDE),
        ], style={"flex": "1 1 50%", "minWidth": "0", "paddingRight": "16px"}),
        html.Div([
            dcc.Graph(
                id="graph",
                figure=fig,
                style={"height": "82vh", "width": "100%"},
                config={
                    "doubleClick": "reset",
                    "displaylogo": False,
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "responsive": True,
                    "modeBarButtonsToAdd": ["lasso2d", "select2d"]
                }
            )
        ], style={"flex": "1 1 50%", "minWidth": "0", "minHeight": "82vh"}),
    ], style={"display": "flex", "alignItems": "stretch", "width": "100%"}),
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
    options = []
    for key in drop_down_dic.keys():
        spec = ANNO_LAYERS.get(key)
        options.append({"label": spec["label"] if spec else key, "value": key})
    return options

@app.callback(
    Output('input_file', 'options'),
    Output('input_file', 'value'),
    Output('output_file', 'options'),
    Input('working_directory', 'value'),
    State('input_file', 'value'),
)
def update_input_file_dropdown(directory, current_file):
    options = list_data_files(directory)
    if current_file in options:
        selected = current_file
    elif options:
        selected = options[0]
    else:
        selected = None
    return options, selected, options

@app.callback(
    Output('output_file', 'value'),
    Input('working_directory', 'value'),
    Input('input_file', 'value'),
)
def sync_output_file(directory, filename):
    if not directory or not filename:
        return None
    return filename

@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("import_data_button", "n_clicks"),
    State("working_directory", "value"),
    State("input_file", "value"),
    running=[
        (Output("import_data_button", "style"), IMPORT_BUSY, IMPORT_IDLE),
        (Output("import_data_button", "disabled"), True, False),
        (Output("io_status", "style", allow_duplicate=True), IO_STATUS_ON, IO_STATUS_OFF),
        (Output("io_status_text", "children", allow_duplicate=True), "Loading sample…", ""),
    ],
    prevent_initial_call=True,
)
def import_sample(_n_clicks, working_directory_val, input_file_val):
    global cell_wall_image, cell_segm_image, df, adata, state_dic, drop_down_dic, state_dic_temp
    if working_directory_val and input_file_val:
        cell_wall_image, cell_segm_image, df, adata, state_dic, drop_down_dic, state_dic_temp = prepare_data(
            os.path.join(working_directory_val, input_file_val)
        )
    return fig

@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("plot_data_button", "n_clicks"),
    State("drop-down_anno_id", "value"),
    State("graph", "relayoutData"),
    running=[
        (Output("plot_data_button", "style"), PLOT_BUSY, PLOT_IDLE),
        (Output("plot_data_button", "disabled"), True, False),
        (Output("io_status", "style", allow_duplicate=True), IO_STATUS_ON, IO_STATUS_OFF),
        (Output("io_status_text", "children", allow_duplicate=True), "Plotting…", ""),
    ],
    prevent_initial_call=True,
)
def plot_loaded_data(_n_clicks, anno_layer, relayout_data):
    global cell_wall_image, df, fig, state_dic, selection_mode
    cell_wall_image, df, fig = plot_data(fig, df, cell_wall_image, state_dic, 0, anno_layer=anno_layer)
    fig.update_layout(dragmode=('lasso' if selection_mode == "lasso" else 'zoom'))
    if isinstance(relayout_data, dict) and fig:
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            fig['layout']['xaxis']['range'] = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
        if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
            fig['layout']['yaxis']['range'] = [relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]
    return fig

@app.callback(
    Output(component_id="graph", component_property="figure", allow_duplicate=True),
    State(component_id='working_directory', component_property='value'),
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
    State(component_id='output_file', component_property='value'),
    Input(component_id='select_click_mode', component_property='n_clicks'),
    Input(component_id='select_brush_mode', component_property='n_clicks'),
    Input(component_id='select_lasso_mode', component_property='n_clicks'),
    State(component_id='brush_radius_px', component_property='value'),
    prevent_initial_call=True
)
def update_plot(working_directory_val,
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

    if triggered_id == "select_click_mode":
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
        if pattern_name and pattern_val:
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
        if working_directory_val and out_path:
            save_data(
                adata, df, state_dic, cell_wall_image, cell_segm_image,
                os.path.join(working_directory_val, out_path),
                drop_down_dic,
            )

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

@app.callback(
    Output("panel_read_write", "style"),
    Output("panel_rotation_cutting", "style"),
    Output("panel_annotation", "style"),
    Output("tab_read_write", "style"),
    Output("tab_rotation_cutting", "style"),
    Output("tab_annotation", "style"),
    Input("tab_read_write", "n_clicks"),
    Input("tab_rotation_cutting", "n_clicks"),
    Input("tab_annotation", "n_clicks"),
)
def switch_left_tabs(n_read_write, n_rotation, n_annotation):
    triggered = ctx.triggered_id
    if triggered == "tab_rotation_cutting":
        return PANEL_HIDE, PANEL_SHOW, PANEL_HIDE, TAB_INACTIVE, TAB_ACTIVE, TAB_INACTIVE
    if triggered == "tab_annotation":
        return PANEL_HIDE, PANEL_HIDE, PANEL_SHOW, TAB_INACTIVE, TAB_INACTIVE, TAB_ACTIVE
    return PANEL_SHOW, PANEL_HIDE, PANEL_HIDE, TAB_ACTIVE, TAB_INACTIVE, TAB_INACTIVE

def _menu_is_open(style):
    return isinstance(style, dict) and style.get("display") != "none"

def _clicked(triggered_id):
    if not ctx.triggered:
        return False
    value = ctx.triggered[0].get("value")
    return bool(value)

@app.callback(
    Output("pattern_name_menu", "style"),
    Output("pattern_val_menu", "style"),
    Input("pattern_name_display", "n_clicks"),
    Input("pattern_val_display", "n_clicks"),
    Input("pattern_name_plus", "n_clicks"),
    Input("pattern_val_plus", "n_clicks"),
    Input({"type": "layer-pick", "index": ALL}, "n_clicks"),
    Input({"type": "val-pick", "index": ALL}, "n_clicks"),
    State("pattern_name_menu", "style"),
    State("pattern_val_menu", "style"),
    prevent_initial_call=True,
)
def toggle_annotation_menus(n_name, n_val, n_name_plus, n_val_plus, layer_picks, val_picks, name_style, val_style):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) or triggered in ("pattern_name_plus", "pattern_val_plus"):
        return MENU_CLOSED, MENU_CLOSED
    if triggered == "pattern_name_display":
        return (MENU_CLOSED if _menu_is_open(name_style) else MENU_OPEN), MENU_CLOSED
    if triggered == "pattern_val_display":
        return MENU_CLOSED, (MENU_CLOSED if _menu_is_open(val_style) else MENU_OPEN)
    return MENU_CLOSED, MENU_CLOSED

def _typing_mode(plus_id):
    if ctx.triggered_id == plus_id:
        return TYPE_CLOSED, TYPE_OPEN, ""
    return TYPE_OPEN, TYPE_CLOSED, ""

@app.callback(
    Output("pattern_name_display", "style"),
    Output("pattern_name_type", "style"),
    Output("pattern_name_type", "value"),
    Input("pattern_name_plus", "n_clicks"),
    Input("pattern_name_type", "n_submit"),
    Input({"type": "layer-pick", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def toggle_annotation_name_typing(n_plus, n_submit, picks):
    return _typing_mode("pattern_name_plus")

@app.callback(
    Output("pattern_val_display", "style"),
    Output("pattern_val_type", "style"),
    Output("pattern_val_type", "value"),
    Input("pattern_val_plus", "n_clicks"),
    Input("pattern_val_type", "n_submit"),
    Input({"type": "val-pick", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def toggle_annotation_id_typing(n_plus, n_submit, picks):
    return _typing_mode("pattern_val_plus")

@app.callback(
    Output("pattern_name", "value"),
    Output("pattern_name_display", "children"),
    Output("pattern_name_menu", "children"),
    Input("pattern_name_type", "n_submit"),
    Input({"type": "layer-pick", "index": ALL}, "n_clicks"),
    Input({"type": "layer-remove", "index": ALL}, "n_clicks"),
    State("pattern_name_type", "value"),
    State("pattern_name", "value"),
    prevent_initial_call=True,
)
def edit_annotation_layers(n_submit, picks, removes, typed, current):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and not _clicked(triggered):
        raise PreventUpdate
    if isinstance(triggered, dict) and triggered.get("type") == "layer-remove":
        if triggered["index"] in BUILTIN_LAYERS:
            raise PreventUpdate
        ANNO_LAYERS.pop(triggered["index"], None)
        if current == triggered["index"]:
            current = next(iter(ANNO_LAYERS), "")
    elif isinstance(triggered, dict) and triggered.get("type") == "layer-pick":
        current = triggered["index"]
    elif triggered == "pattern_name_type":
        text = (typed or "").strip()
        if not text:
            raise PreventUpdate
        current = ensure_annotation_layer(text)
    else:
        raise PreventUpdate
    label = ANNO_LAYERS.get(current, {}).get("label", current or "Select")
    return current, label, option_rows("layer", layer_dropdown_options(), current)

@app.callback(
    Output("pattern_val", "value"),
    Output("pattern_val_display", "children"),
    Output("pattern_val_menu", "children"),
    Input("pattern_name", "value"),
    Input("pattern_val_type", "n_submit"),
    Input({"type": "val-pick", "index": ALL}, "n_clicks"),
    Input({"type": "val-remove", "index": ALL}, "n_clicks"),
    State("pattern_val_type", "value"),
    State("pattern_val", "value"),
    prevent_initial_call=True,
)
def edit_annotation_ids(layer_key, n_submit, picks, removes, typed, current):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and not _clicked(triggered):
        raise PreventUpdate
    if layer_key not in ANNO_LAYERS:
        return "", "Select", []
    values = ANNO_LAYERS[layer_key]["values"]
    if isinstance(triggered, dict) and triggered.get("type") == "val-remove":
        victim = triggered["index"]
        if victim in BUILTIN_VALUES.get(layer_key, set()):
            raise PreventUpdate
        if victim in values:
            values.remove(victim)
        if current == victim:
            current = values[0] if values else ""
    elif isinstance(triggered, dict) and triggered.get("type") == "val-pick":
        current = triggered["index"]
    elif triggered == "pattern_val_type":
        text = (typed or "").strip()
        if not text:
            raise PreventUpdate
        if text not in values:
            values.append(text)
        current = text
    elif triggered == "pattern_name":
        if current not in values:
            current = values[0] if values else ""
    else:
        raise PreventUpdate
    options = [{"label": value, "value": value} for value in values]
    return current, (current or "Select"), option_rows("val", options, current)

app.clientside_callback(
    """
    function(nClicks) {
        if (!nClicks) {
            return window.dash_clientside.no_update;
        }
        let tries = 0;
        const timer = setInterval(function() {
            const el = document.getElementById("pattern_name_type");
            tries += 1;
            if (el && el.offsetParent !== null) {
                el.focus();
                clearInterval(timer);
            }
            if (tries > 20) clearInterval(timer);
        }, 50);
        return Date.now();
    }
    """,
    Output("pattern_name_focus", "data"),
    Input("pattern_name_plus", "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(nClicks) {
        if (!nClicks) {
            return window.dash_clientside.no_update;
        }
        let tries = 0;
        const timer = setInterval(function() {
            const el = document.getElementById("pattern_val_type");
            tries += 1;
            if (el && el.offsetParent !== null) {
                el.focus();
                clearInterval(timer);
            }
            if (tries > 20) clearInterval(timer);
        }, 50);
        return Date.now();
    }
    """,
    Output("pattern_val_focus", "data"),
    Input("pattern_val_plus", "n_clicks"),
    prevent_initial_call=True,
)

if __name__ == '__main__':
    app.run(debug=True, port=8051)
