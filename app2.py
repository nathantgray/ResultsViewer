#! venv/bin/python
import csv
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import plotly.graph_objs as go
import io
from pathlib import Path
from drawio2cytoscape import drawio2cytoscape, get_pages
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import base64
import matplotlib.colors as mpcolor
from ast import literal_eval
from matplotlib import cm


class Diagram:
    def __init__(self, diagram_path):
        diagram_path = Path(diagram_path)
        self.tree = ET.parse(diagram_path)
        self.elements = drawio2cytoscape(self.tree)
        self.page_list = get_pages(self.tree)

    # def set_tree(self, tree):
    #     self.tree = tree
    #
    # def set_elements(self, elements):
    #     self.elements = elements
    #
    # def set_page_list(self, page_list):
    #     self.page_list = page_list
    #
    # def get_tree(self):
    #     return self.tree
    #
    # def get_elements(self):
    #     return self.elements
    #
    # def get_page_list(self):
    #     return self.page_list


app = dash.Dash(external_stylesheets=[dbc.themes.YETI])

# colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}
PATH = Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# Define elements, stylesheet and layout
diagram = Diagram(Path('./data/ieee123.drawio'))

figure = go.Figure(
    data=[
        go.Scatter(
            x=np.array([]),
            y=np.array([]),
            mode='lines+markers')
    ],
    layout=go.Layout(
        height=250,
        margin={'t': 5, 'l': 5, 'b': 5, 'r': 5},
        # plot_bgcolor=colors["graphBackground"],
        # paper_bgcolor=colors["graphBackground"]
    ))

df_array = [[None, None, None], [None, None, None], [None, None, None]]
# node_prefix = 'node_'
default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            "text-valign": "top",
            "text-halign": "left",
        }
    }
]

setup_settings = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id='page_list',
                options=diagram.page_list,
                value=list(diagram.page_list[0].values())[0], persistence=True, persistence_type='local'
            ),
        ),
        dbc.Col(
            html.Div(id='node_prefix_input_box_label', children='Node prefix:')
        ),
        dbc.Col(
            dbc.Input(id="node_prefix", placeholder="node_", value="node_", persistence=True, persistence_type='local')
        )
    ]
)

limit_settings = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id='phase_selector',
                options=[
                    {'label': "none", 'value': 0},
                    {'label': "Data Set #1", 'value': 1},
                    {'label': "Data Set #2", 'value': 2},
                    {'label': "Data Set #3", 'value': 3},

                ],
                value=0, persistence=True, persistence_type='local'
            ),
        ),
        dbc.Col(
            dbc.Input(id="up_limit", placeholder="Scale Max", type="number", persistence=True, persistence_type='local')
        ),
        dbc.Col(
            dbc.Input(id="low_limit", placeholder="Scale Min", type="number", persistence=True,
                      persistence_type='local')
        ),
    ]
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            html.Div(
                                # className="header",
                                children=[
                                    html.Div(
                                        className="div-info",
                                        children=[
                                            html.H2(
                                                # className="title",
                                                children="GridLAB-D Results Viewer"),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                        dbc.Row(
                            [
                                # dbc.Col(dbc.Row(html.H4("Network")), width='auto'),
                                dbc.Col(dbc.Card(
                                    [
                                        dcc.Upload(
                                            id='import_diagram',
                                            children=html.Div(['Select Diagram: ', html.A('file')]),
                                            # Allow multiple files to be uploaded
                                            multiple=False
                                        ),
                                    ]
                                ), width='auto'),
                            ]
                        ),
                        setup_settings,
                        limit_settings,
                        cyto.Cytoscape(
                            id="cytoscape",
                            elements=diagram.elements,
                            stylesheet=default_stylesheet,
                            layout={"name": "preset", "fit": True, "animate": True},
                            style={
                                "height": "650px",
                                "width": "100%",
                                # "backgroundColor": "white",
                                "margin": "auto",
                            },
                            minZoom=0.35,
                        ),

                        dcc.Slider(
                            id='slider',
                            marks={i: '{}'.format(10 ** i) for i in range(4)},
                            max=60,
                            value=0,
                            step=1,
                            updatemode='drag'
                        ),
                        html.Div(id='slider-output')
                    ],
                    width={'size': 5, 'offset': 0}
                ),
                dbc.Col(
                    [
                        dbc.Row(

                            dcc.Upload(
                                id='import_node_data',
                                children=html.Div(['Node Data: ', html.A('Select file for each phase')]),
                                multiple=True
                            ),
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Input(id="base1", placeholder="base", type="number", value=2401.7771198288433,
                                              persistence=True, persistence_type='local')),
                                dbc.Col(dbc.Input(id="up_limit1", placeholder="max", type="number", value=1.05,
                                                  persistence=True, persistence_type='local')),
                                dbc.Col(dbc.Input(id="low_limit1", placeholder="min", type="number", value=0.95,
                                                  persistence=True, persistence_type='local')),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="node_graph1",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="node_graph2",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="node_graph3",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                    ],
                    # width="2"
                ),
                dbc.Col(
                    [
                        dbc.Row(

                            dcc.Upload(
                                id='import_edge_data1',
                                children=html.Div(['Line Data: ', html.A('Select file for each phase')]),
                                multiple=True
                            ),
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Input(id="base_edge1", placeholder="base", type="number", persistence=True,
                                              persistence_type='local')
                                ),
                                dbc.Col(
                                    dbc.Input(id="up_limit_edge1", placeholder="max", type="number", persistence=True,
                                              persistence_type='local')
                                ),
                                dbc.Col(
                                    dbc.Input(id="low_limit_edge1", placeholder="min", type="number", persistence=True,
                                              persistence_type='local')
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="edge_graph1",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="edge_graph2",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="edge_graph3",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                    ],
                    # width="2"
                ),
                dbc.Col(
                    [
                        dbc.Row(

                            dcc.Upload(
                                id='import_edge_data2',
                                children=html.Div(['Line Data: ', html.A('Select file for each phase')]),
                                multiple=True
                            ),
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Input(id="base_edge4", placeholder="base", type="number", persistence=True,
                                              persistence_type='local')
                                ),
                                dbc.Col(
                                    dbc.Input(id="up_limit_edge4", placeholder="max", type="number", persistence=True,
                                              persistence_type='local')
                                ),
                                dbc.Col(
                                    dbc.Input(id="low_limit_edge4", placeholder="min", type="number", persistence=True,
                                              persistence_type='local')
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="edge_graph4",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="edge_graph5",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Graph(
                                            id="edge_graph6",
                                            figure=figure,
                                            # config={'frameMargins': 0}
                                        )
                                    ],
                                    className='w-100'
                                ),
                            ]
                        ),
                    ],
                    # width="2"
                ),
            ]
        ),
    ],
    fluid=True,
)


def weights_to_colors(weights, cmap):
    # from https://community.plotly.com/t/how-to-scale-node-colors-in-cytoscape/23176/3
    # weights is the list of node weights
    # cmap a mpl colormap
    colors01 = cmap(weights)
    colors01 = np.array([c[:3] for c in colors01])
    colors255 = (255 * colors01 + 0.5).astype(np.uint8)
    hexcolors = [f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in colors255]
    return hexcolors


def color_for_val(val, vmin, vmax, pl_colorscale):
    # val: float to be mapped to the Plotlt colorscale
    # [vmin, vmax]: the range of val values
    # pl_colorscale is a Plotly colorscale with colors in the RGB space with R,G, B in  0-255
    # this function maps the normalized value of val to a color in the colorscale
    if vmin >= vmax:
        raise ValueError('vmin must be less than vmax')

    scale = [item[0] for item in pl_colorscale]
    plotly_colors = [item[1] for item in pl_colorscale]  # i.e. list of 'rgb(R, G, B)'

    colors_01 = np.array([literal_eval(c[3:]) for c in plotly_colors]) / 255  # color codes in [0,1]

    v = (val - vmin) / (vmax - vmin)  # val is mapped to v in [0,1]
    # find two consecutive values, left and right, in scale such that   v  lie within  the corresponding interval
    idx = np.digitize(v, scale)
    left = scale[idx - 1]
    right = scale[idx]

    vv = (v - left) / (right - left)  # normalize v with respect to [left, right]

    # get   [0,1]-valued, 0-255, and hex color code for the  color associated to  val
    vcolor01 = colors_01[idx - 1] + vv * (colors_01[idx] - colors_01[idx - 1])  # linear interpolation
    vcolor255 = (255 * vcolor01 + 0.5).astype(np.uint8)
    hexcolor = f'#{vcolor255[0]:02x}{vcolor255[1]:02x}{vcolor255[2]:02x}'
    # for dash-cytoscale we need the hex representation:
    return hexcolor


@app.callback(Output('slider-output', 'children'),
              Input('slider', 'value'))
def display_value(value):
    return f'{value}'


@app.callback(
    Output("page_list", "options"),
    [Input("import_diagram", "contents")]
)
def upload_diagram(contents):
    if contents is not None:
        decoded = base64.b64decode(contents.split(',')[1])
        diagram.tree = ET.ElementTree(ET.fromstring(decoded))
        page_list = get_pages(diagram.tree)
        return page_list
    else:
        return diagram.page_list


@app.callback(
    Output('import_diagram', 'children'),
    [
        Input('import_diagram', 'filename'),
    ]
)
def update_upload_text(filename):
    if filename is not None:
        return html.Div(['Select Diagram: ', html.A(filename)])
    else:
        return html.Div(['Select Diagram: ', html.A('file')])


@app.callback(
    Output("cytoscape", "elements"),
    [Input("page_list", "value")]
)
def show_page(page):
    return drawio2cytoscape(diagram.tree, page=page)


@app.callback(Output('slider', 'max'),
              [Input('phase_selector', 'value')])
def update_slider_range(phase_selector):
    global df_array
    length = 0
    if 0 < phase_selector < 4:
        i = phase_selector - 1
        if df_array[0][i] is not None:
            if df_array[0][i].shape[0] > length:
                length = df_array[0][i].shape[0]
        if df_array[1][i] is not None:
            if df_array[1][i].shape[0] > length:
                length = df_array[1][i].shape[0]
        if df_array[2][i] is not None:
            if df_array[2][i].shape[0] > length:
                length = df_array[2][i].shape[0]

    return length


@app.callback(Output('cytoscape', 'stylesheet'),
              [Input('slider', 'value'),
               Input('phase_selector', 'value'),
               Input('node_prefix', 'value'),
               Input('up_limit', 'value'),
               Input('low_limit', 'value'),
               Input('base1', 'value'),
               Input('up_limit1', 'value'),
               Input('low_limit1', 'value'),

               Input('base_edge1', 'value'),
               Input('up_limit_edge1', 'value'),
               Input('low_limit_edge1', 'value'),

               Input("page_list", "value")],
              prevent_initial_callbacks=True
              )
def update_stylesheet(t, phase_selector, node_prefix, vmax, vmin,
                      base, up_limit, low_limit,
                      base_edge, up_limit_edge, low_limit_edge,
                      page):
    # global node_prefix
    global df_array
    new_styles = []

    if t is None:
        t = 0
    flow_max = None
    if node_prefix is None:
        node_prefix = 'node_'

    # cmap = px.colors.sequential.solar

    if 0 < phase_selector < 4:
        df = df_array[0][phase_selector - 1]
        df_edge1 = df_array[1][phase_selector - 1]
        df_edge2 = df_array[2][phase_selector - 1]

        if base is None:
            base = 1
        if low_limit is None:
            low_limit = -1e9
        if up_limit is None:
            up_limit = 1e9
        if base_edge is None:
            base_edge = 1
        if low_limit_edge is None:
            low_limit_edge = -1e9
        if up_limit_edge is None:
            up_limit_edge = 1e9
        if df is not None:
            if vmax is None:
                vmax = np.max(np.array([df[key].max() for key in df.keys() if df[key].max() is not None])) / base
            if vmin is None:
                vmin = np.min(np.array([df[key].min() for key in df.keys() if df[key].min() > 0])) / base
            norm = mpcolor.Normalize(vmin, vmax)
            cmap = cm.get_cmap('viridis', 512)
            if t >= df.shape[0]:
                t = df.shape[0] - 1
            for key in df.keys():
                if df[key][t] / base > up_limit:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace(node_prefix, "")}"]',
                            'style': {
                                'shape': 'triangle',
                                'border-width': 3,
                                'border-style': 'double',
                                'border-color': 'red',
                                'background-color': mpcolor.to_hex(cmap(norm(df[key][t] / base)))

                            }
                        }
                    )
                elif df[key][t] / base < low_limit and df[key].min() != 0:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace(node_prefix, "")}"]',
                            'style': {
                                'shape': 'square',
                                'border-width': 3,
                                'border-style': 'double',
                                'border-color': 'red',
                                'background-color': mpcolor.to_hex(cmap(norm(df[key][t] / base)))
                            }
                        }
                    )
                elif low_limit < df[key][t] / base < up_limit:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace(node_prefix, "")}"]',
                            'style': {
                                'shape': 'ellipse',
                                'border-width': 3,
                                'border-color': 'black',
                                'border-style': 'solid',
                                'background-color': mpcolor.to_hex(cmap(norm(df[key][t] / base)))
                            }
                        }
                    )
                if df[key].max() == 0:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace(node_prefix, "")}"]',
                            'style': {
                                'shape': 'ellipse',
                                # 'border-width': 1,
                                # 'border-color': 'black',
                                # 'border-style': 'dotted',
                                # 'backgound-color': 'white'
                            }
                        }
                    )
        if df_edge1 is not None:
            if flow_max is None:
                flow_max = np.max(np.array(
                    [df_edge1[key].max() for key in df_edge1.keys() if df_edge1[key].max() is not None])) / base_edge
            for edge_key in df_edge1.keys():
                # print(edge_key)
                p_flow = df_edge1[edge_key][t] / base_edge
                q_flow = df_edge2[edge_key][t] / base_edge
                scale = 4 * np.sqrt(np.abs(p_flow) / flow_max)
                # edge_string = edge_key.replace('oh_line_', '')
                from_to = edge_key.split('_')[-2:]
                from_value = from_to[0]  # int(from_to[0])
                to_value = from_to[1]  # int(from_to[1])
                try:
                    from_id = diagram.tree.find(f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@vertex='1'][@value='{from_value}']").attrib.get('id')
                    to_id = diagram.tree.find(f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@vertex='1'][@value='{to_value}']").attrib.get('id')
                    edge_mx = diagram.tree.find(f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@target='{to_id}']")
                    if scale > 1e-6:
                        if edge_mx is None or edge_mx.attrib.get('source') != from_id:
                            # edge direction is reversed
                            edge_mx = diagram.tree.find(
                                f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@target='{from_id}']")
                            to_id = diagram.tree.find(
                                f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@vertex='1'][@value='{from_value}']").attrib.get(
                                'id')
                            p_flow = -p_flow
                            q_flow = -q_flow

                        target = 'target'
                        if p_flow < 0:
                            target = 'source'
                        new_styles.append(
                            {
                                # 'selector': f'edge[id="{edge_mx.attrib.get("id")}"]',
                                'selector': f"[target = '{to_id}']",
                                'style': {
                                    f'mid-{target}-arrow-shape': 'vee',
                                    f'mid-{target}-arrow-color': 'blue',
                                    'arrow-scale': f'{scale}',
                                    'line-color': 'black',
                                    f'{target}-label': f'{round(p_flow, 3) + 1j*round(q_flow, 3)}',
                                    f'{target}-text-rotation': 'autorotate',
                                    'text-wrap': 'wrap',
                                    'font-size': '5',
                                    'text-background-color': 'yellow',
                                    'text-background-opacity': '1',
                                    f'{target}-text-offset': '32',
                                    'text-max-width': '1000px',

                                }
                            }
                        )
                        
                except KeyError:
                    continue
                except SyntaxError:
                    continue
                # print(edge_key)

    return default_stylesheet + new_styles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ Update Plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def update_graph(df, nodedata, base, node_prefix):
    if base is None:
        base = 1
    if node_prefix is None:
        node_prefix = 'node_'
    if nodedata is not None and df is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                y = np.array(df[f'{node_prefix}{node_name}']) / base
                x = np.array(range(0, len(y)))
                fig_data.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=node_name,
                        mode='lines+markers')
                )
            return go.Figure(
                fig_data,
                layout=go.Layout(
                    height=250,
                    margin={'t': 5, 'l': 5, 'b': 5, 'r': 5},
                    # plot_bgcolor=colors["graphBackground"],
                    # paper_bgcolor=colors["graphBackground"],
                    showlegend=True
                )
            )
    return figure


def update_edge_graph(df, edgedata, base, page):
    if base is None:
        base = 1
    if edgedata is not None and df is not None:
        if len(edgedata) > 0:
            fig_data = []
            for data in edgedata:
                from_bus = diagram.tree.find(
                    f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@id='{data['source']}']").attrib.get('value')
                to_bus = diagram.tree.find(
                    f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@id='{data['target']}']").attrib.get('value')

                edge_key = None
                for key in df.keys():
                    if f'{from_bus}_{to_bus}' in key:
                        edge_key = key
                        break

                y = np.array(df[edge_key]) / base
                x = np.array(range(0, len(y)))
                fig_data.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=f'{from_bus}-{to_bus}',
                        mode='lines+markers')
                )
            return go.Figure(
                fig_data,
                layout=go.Layout(
                    height=250,
                    margin={'t': 5, 'l': 5, 'b': 5, 'r': 5},
                    showlegend=True
                )
            )
    return figure


@app.callback(
    Output('node_graph1', 'figure'), [
        Input("cytoscape", "selectedNodeData"),
        Input('base1', 'value'),
        Input('node_prefix', 'value')
    ]
)
def update_graph1(nodedata, base, node_prefix):
    return update_graph(df_array[0][0], nodedata, base, node_prefix)


@app.callback(
    Output('node_graph2', 'figure'), [
        Input("cytoscape", "selectedNodeData"),
        Input('base1', 'value'),
        Input('node_prefix', 'value')
    ]
)
def update_graph2(nodedata, base, node_prefix):
    return update_graph(df_array[0][1], nodedata, base, node_prefix)


@app.callback(
    Output('node_graph3', 'figure'), [
        Input("cytoscape", "selectedNodeData"),
        Input('base1', 'value'),
        Input('node_prefix', 'value')
    ]
)
def update_graph3(nodedata, base, node_prefix):
    return update_graph(df_array[0][2], nodedata, base, node_prefix)


@app.callback(
    Output('edge_graph1', 'figure'), [
        Input("cytoscape", "selectedEdgeData"),
        Input('base_edge1', 'value'),
        Input("page_list", "value")

    ]
)
def update_edge_graph1(edgedata, base, page):
    return update_edge_graph(df_array[1][0], edgedata, base, page)


@app.callback(
    Output('edge_graph2', 'figure'), [
        Input("cytoscape", "selectedEdgeData"),
        Input('base_edge1', 'value'),
        Input("page_list", "value")
    ]
)
def update_edge_graph2(edgedata, base, page):
    return update_edge_graph(df_array[1][1], edgedata, base, page)


@app.callback(
    Output('edge_graph3', 'figure'), [
        Input("cytoscape", "selectedEdgeData"),
        Input('base_edge1', 'value'),
        Input("page_list", "value")
    ]
)
def update_edge_graph3(edgedata, base, page):
    return update_edge_graph(df_array[1][2], edgedata, base, page)


@app.callback(
    Output('edge_graph4', 'figure'), [
        Input("cytoscape", "selectedEdgeData"),
        Input('base_edge4', 'value'),
        Input("page_list", "value")

    ]
)
def update_edge_graph1(edgedata, base, page):
    return update_edge_graph(df_array[2][0], edgedata, base, page)


@app.callback(
    Output('edge_graph5', 'figure'), [
        Input("cytoscape", "selectedEdgeData"),
        Input('base_edge4', 'value'),
        Input("page_list", "value")
    ]
)
def update_edge_graph2(edgedata, base, page):
    return update_edge_graph(df_array[2][1], edgedata, base, page)


@app.callback(
    Output('edge_graph6', 'figure'), [
        Input("cytoscape", "selectedEdgeData"),
        Input('base_edge4', 'value'),
        Input("page_list", "value")
    ]
)
def update_edge_graph3(edgedata, base, page):
    return update_edge_graph(df_array[2][2], edgedata, base, page)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ Import Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@app.callback(
    Output('import_node_data', 'children'),
    [
        Input('import_node_data', 'contents'),
        Input('import_node_data', 'filename'),
    ]
)
def import_node_data(files, filenames):
    return import_plot_data(files, filenames, 0)


@app.callback(
    Output('import_edge_data1', 'children'),
    [
        Input('import_edge_data1', 'contents'),
        Input('import_edge_data1', 'filename'),
    ]
)
def import_edge_data1(files, filenames):
    return import_plot_data(files, filenames, 1)


@app.callback(
    Output('import_edge_data2', 'children'),
    [
        Input('import_edge_data2', 'contents'),
        Input('import_edge_data2', 'filename'),
    ]
)
def import_edge_data2(files, filenames):
    return import_plot_data(files, filenames, 2)


def import_plot_data(files, filenames, plot_col_index):
    show_string = 'Select file for each phase'
    if files is not None:
        global df_array
        for filename, file in zip(filenames, files):
            show_string = ''
            if 'A' in filename:
                df_array[plot_col_index][0] = parse_data(file, filename)
                show_string + filename + ' '
            if 'B' in filename:
                df_array[plot_col_index][1] = parse_data(file, filename)
                show_string + filename + ' '
            if 'C' in filename:
                df_array[plot_col_index][2] = parse_data(file, filename)
                show_string + filename + ' '
        print(show_string)
        return html.Div(['Data: ', html.A(show_string)])
    else:
        return html.Div(['Data: ', html.A('Select file for each phase')])


def get_delimiter_and_header(file_path, n_bytes=4000):
    with open(file_path, "r") as csvfile:
        data = csvfile.read(n_bytes)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(data).delimiter

        has_header = sniffer.has_header(data)
        csvfile.seek(0)
        header = None
        if has_header:
            header = 0
        return delimiter, header


def file_importer(data_path, col_names):
    delimiter, header = get_delimiter_and_header(file_path=data_path)
    names = None
    if header is None:
        names = col_names
    return pd.read_csv(data_path, sep=delimiter, header=header, names=names, index_col=False)


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")
    df_import = None
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df_import = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=',', header=8, index_col=0,
                                    parse_dates=True)
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df_import = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df_import = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df_import


if __name__ == "__main__":
    app.run_server(debug=True)
