import math
import dash
import plotly.express as px
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
import zlib
import matplotlib.colors as mpcolor
from ast import literal_eval
import plotly.express as px
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

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

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
global df1
global df2
global df3
default_stylesheet = []
# stylesheet = [
#     {
#         "selector": ".nonterminal",
#         "style": {
#             "label": "data(confidence)",
#             "background-opacity": 0,
#             "text-halign": "left",
#             "text-valign": "top",
#         },
#     },
#     {"selector": ".support", "style": {"background-opacity": 0}},
#     {
#         "selector": "edge",
#         "style": {
#             "source-endpoint": "inside-to-node",
#             "target-endpoint": "inside-to-node",
#         },
#     },
#     {
#         "selector": ".terminal",
#         "style": {
#             "label": "data(name)",
#             "width": 10,
#             "height": 10,
#             "text-valign": "center",
#             "text-halign": "right",
#             "background-color": "#222222",
#         },
#     },
# ]

limit_settings = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id='limits',
                options=[
                    {'label': "none", 'value': 0},
                    {'label': "Data Set #1", 'value': 1},
                    {'label': "Data Set #2", 'value': 2},
                    {'label': "Data Set #3", 'value': 3},

                ],
                value=0
            ),
        ),
        dbc.Col(
            dbc.Input(id="up_limit", placeholder="Scale Max", type="number")
        ),
        dbc.Col(
            dbc.Input(id="low_limit", placeholder="Scale Min", type="number")
        ),
    ]
)

graph_settings1 = dbc.Row(
    [
        dbc.Col(
            dcc.Upload(
                id='import_data1',
                children=html.Div(['Data: ', html.A('select')]),
                multiple=False
            ),
        ),
        dbc.Col(
            dbc.Input(id="base1", placeholder="Base", type="number")
        ),
        dbc.Col(
            dbc.Input(id="up_limit1", placeholder="Upper Limit (p.u.)", type="number")
        ),
        dbc.Col(
            dbc.Input(id="low_limit1", placeholder="Lower Limit (p.u.)", type="number")
        ),
    ]
)
graph_settings2 = dbc.Row(
    [
        dbc.Col(
            dcc.Upload(
                id='import_data2',
                children=html.Div(['Data: ', html.A('select')]),
                multiple=False
            ),
        ),
        dbc.Col(
            dbc.Input(id="base2", placeholder="Base", type="number")
        ),
        dbc.Col(
            dbc.Input(id="up_limit2", placeholder="Upper Limit (p.u.)", type="number")
        ),
        dbc.Col(
            dbc.Input(id="low_limit2", placeholder="Lower Limit (p.u.)", type="number")
        ),
    ]
)

graph_settings3 = dbc.Row(
    [
        dbc.Col(
            dcc.Upload(
                id='import_data3',
                children=html.Div(['Data: ', html.A('select')]),
                multiple=False
            ),
        ),
        dbc.Col(
            dbc.Input(id="base3", placeholder="Base", type="number")
        ),
        dbc.Col(
            dbc.Input(id="up_limit3", placeholder="Upper Limit (p.u.)", type="number")
        ),
        dbc.Col(
            dbc.Input(id="low_limit3", placeholder="Lower Limit (p.u.)", type="number")
        ),
    ]
)
app.layout = html.Div(
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
                                            # html.P(
                                            #     """
                                            #     Dash Cytoscape is a graph visualization component for creating easily customizable,
                                            #     high-performance interactive, and web-based networks.
                                            #     """
                                            # ),
                                            # html.A(
                                            #     children=html.Button("Run Model", id='run_model', className="button"),
                                            #     # href="https://www.gridlabd.org/index.stm",
                                            #     target="_blank",
                                            # ),
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
                        dcc.Dropdown(
                            id='page_list',
                            options=diagram.page_list,
                            value=list(diagram.page_list[0].values())[0]
                        ),
                        limit_settings,
                        cyto.Cytoscape(
                            id="cytoscape",
                            elements=diagram.elements,
                            # stylesheet=stylesheet,
                            layout={"name": "preset", "fit": True, "animate": True},
                            style={
                                "height": "650px",
                                "width": "100%",
                                # "backgroundColor": "white",
                                "margin": "auto",
                            },
                            minZoom=0.3,
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
                    width={'size': 5, 'offset': 1}
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        graph_settings1,
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
                                        graph_settings2,
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
                                        graph_settings3,
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
                    width="5"
                ),
            ]
        ),
    ]
)


def weights_to_colors(weights, cmap):
    # from https://community.plotly.com/t/how-to-scale-node-colors-in-cytoscape/23176/3
    # weights is the list of node weights
    # cmap a mpl colormap
    colors01 = cmap(weights)
    colors01 = np.array([c[:3] for c in colors01])
    colors255 = (255*colors01+0.5).astype(np.uint8)
    hexcolors = [f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in colors255]
    return hexcolors


def color_for_val(val, vmin, vmax, pl_colorscale):
    # val: float to be mapped to the Plotlt colorscale
    #[vmin, vmax]: the range of val values
    # pl_colorscale is a Plotly colorscale with colors in the RGB space with R,G, B in  0-255
    # this function maps the normalized value of val to a color in the colorscale
    if vmin >= vmax:
        raise ValueError('vmin must be less than vmax')

    scale = [item[0] for item in pl_colorscale]
    plotly_colors = [item[1] for item in pl_colorscale]# i.e. list of 'rgb(R, G, B)'

    colors_01 = np.array([literal_eval(c[3:]) for c in plotly_colors])/255  #color codes in [0,1]

    v= (val - vmin) / (vmax - vmin) # val is mapped to v in [0,1]
    #find two consecutive values, left and right, in scale such that   v  lie within  the corresponding interval
    idx = np.digitize(v, scale)
    left = scale[idx-1]
    right = scale[idx]

    vv = (v - left) / (right - left) #normalize v with respect to [left, right]

    #get   [0,1]-valued, 0-255, and hex color code for the  color associated to  val
    vcolor01 = colors_01[idx-1] + vv * (colors_01[idx] - colors_01[idx-1])  #linear interpolation
    vcolor255 = (255*vcolor01+0.5).astype(np.uint8)
    hexcolor = f'#{vcolor255[0]:02x}{vcolor255[1]:02x}{vcolor255[2]:02x}'
    #for dash-cytoscale we need the hex representation:
    return hexcolor


@app.callback(Output('slider-output', 'children'),
              Input('slider', 'value'))
def display_value(value):
    return f'{value}'


@app.callback(
    Output("node-data", "children"), [Input("cytoscape", "selectedNodeData")]
)
def display_nodedata(datalist):
    contents = "Click on a node to see its details here"
    if datalist is not None:
        if len(datalist) > 0:
            data = datalist[-1]
            contents = []
            contents.append(html.H5("Node: " + data["label"].title()))
            contents.append(
                html.P(
                    "Loads: "
                    # + str(data["authors"])
                    # + ", Citations: "
                    # + str(data["n_cites"])
                )
            )
            contents.append(
                html.P(
                    "Generation: "
                    # + str(data["authors"])
                    # + ", Citations: "
                    # + str(data["n_cites"])
                )
            )
            contents.append(
                html.P(
                    "Voltage Magnitude: "
                    # + data["journal"].title()
                    # + ", Published: "
                    # + data["pub_date"]
                )
            )
            contents.append(
                html.P(
                    "Voltage Angle: "
                    # + str(data["authors"])
                    # + ", Citations: "
                    # + str(data["n_cites"])
                )
            )

    return contents

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


@app.callback(Output('cytoscape', 'stylesheet'),
              [Input('slider', 'value'),
               Input('limits', 'value'),
               Input('up_limit', 'value'),
               Input('low_limit', 'value'),
               Input('base1', 'value'),
               Input('up_limit1', 'value'),
               Input('low_limit1', 'value'),
               Input('base2', 'value'),
               Input('up_limit2', 'value'),
               Input('low_limit2', 'value'),
               Input('base3', 'value'),
               Input('up_limit3', 'value'),
               Input('low_limit3', 'value')],
                prevent_initial_callbacks=True
              )
def update_stylesheet(t, limits, vmax, vmin,
                      base1, up_limit1, low_limit1,
                      base2, up_limit2, low_limit2,
                      base3, up_limit3, low_limit3):
    new_styles = []
    df = None
    up_limit = None
    low_limit = None
    base = 1
    if t is None:
        t = 0

    # cmap = px.colors.sequential.solar

    if 0 < limits < 4:
        if limits == 1:
            df = df1
            up_limit = up_limit1
            low_limit = low_limit1
            base = base1
        if limits == 2:
            df = df2
            up_limit = up_limit2
            low_limit = low_limit2
            base = base2
        if limits == 3:
            df = df3
            up_limit = up_limit3
            low_limit = low_limit3
            base = base3
        if base is None:
            base = 1
        if low_limit is None:
            low_limit = 0
        if up_limit is None:
            up_limit = 100000000
        if df is not None:
            if vmax is None:
                vmax = np.max(np.array([df[key].max() for key in df.keys() if df[key].max() is not None]))/base
            if vmin is None:
                vmin = np.min(np.array([df[key].min() for key in df.keys() if df[key].min() > 0]))/base
            norm = mpcolor.Normalize(vmin, vmax)
            cmap = cm.get_cmap('viridis', 512)
            for key in df.keys():
                if df[key].max()/base > up_limit:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace("node_", "")}"]',
                            'style': {
                                'shape': 'triangle',
                                'border-width': 3,
                                'border-style': 'double',
                                'border-color': 'red',
                                'background-color': mpcolor.to_hex(cmap(norm(df[key][t]/base)))

                            }
                        }
                    )
                elif df[key].min()/base < low_limit and df[key].min() != 0:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace("node_", "")}"]',
                            'style': {
                                'shape': 'square',
                                'border-width': 3,
                                'border-style': 'double',
                                'border-color': 'red',
                                'background-color': mpcolor.to_hex(cmap(norm(df[key][t]/base)))
                            }
                        }
                    )
                elif df[key].min()/base > low_limit and df[key].max()/base < up_limit:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace("node_", "")}"]',
                            'style': {
                                'shape': 'ellipse',
                                'border-width': 3,
                                'border-color': 'black',
                                'border-style': 'solid',
                                'background-color': mpcolor.to_hex(cmap(norm(df[key][t]/base)))
                            }
                        }
                    )

                if df[key].max() == 0:
                    new_styles.append(
                        {
                            'selector': f'node[label = "{key.replace("node_", "")}"]',
                            'style': {
                                'shape': 'ellipse',
                                # 'border-width': 1,
                                # 'border-color': 'black',
                                # 'border-style': 'dotted',
                                # 'backgound-color': 'white'
                            }
                        }
                    )

    return default_stylesheet + new_styles


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")
    df_import = None
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df_import = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=',', header=8, index_col=0, parse_dates=True)
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


@app.callback(
    Output('node_graph1', 'figure'), [
    Input("cytoscape", "selectedNodeData"),
    Input('base1', 'value')
    ]
)
def update_graph1(nodedata, base1):
    base = 1
    if base1 is not None:
        base = base1
    if nodedata is not None and df1 is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = df1
                y = np.array(df[f'node_{node_name}'])/base
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


@app.callback(
    Output('node_graph2', 'figure'), [
    Input("cytoscape", "selectedNodeData"),
    Input('base2', 'value')
    ]
)
def update_graph2(nodedata, base2):
    base = 1
    if base2 is not None:
        base = base2
    if nodedata is not None and df2 is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = df2
                y = np.array(df[f'node_{node_name}'])/base
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


@app.callback(
    Output('node_graph3', 'figure'), [
    Input("cytoscape", "selectedNodeData"),
    Input('base3', 'value')
    ]
)
def update_graph3(nodedata, base3):
    base = 1
    if base3 is not None:
        base = base3
    if nodedata is not None and df3 is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = df3
                y = np.array(df[f'node_{node_name}'])/base
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


# @app.callback(
#     Output('edge_graph', 'figure'), [
#     Input("cytoscape", "selectedEdgeData")
#     ]
# )
# def update_graph_edge(edge_data):
#     if edge_data is not None and volt_df is not None:
#         if len(edge_data) > 0:
#             fig_data = []
#             for data in edge_data:
#                 edge_name = data["label"]
#                 df = volt_df
#                 y = np.array(df[f'node_{edge_name}'])
#                 x = np.array(range(0, len(y)))
#                 fig_data.append(
#                     go.Scatter(
#                         x=x,
#                         y=y,
#                         name=edge_name,
#                         mode='lines+markers')
#                 )
#             return go.Figure(
#                 fig_data,
#                 layout=go.Layout(
#                     plot_bgcolor=colors["graphBackground"],
#                     paper_bgcolor=colors["graphBackground"],
#                     showlegend=True
#                 )
#             )
#     return figure

@app.callback(
    Output('import_data1', 'children'),
    [
    Input('import_data1', 'contents'),
    Input('import_data1', 'filename'),
    ]
)
def import_data1(contents, filename):
    if contents is not None:
        contents = contents
        filename = filename
        global df1
        df1 = parse_data(contents, filename)
        return html.Div([html.A(filename)])
    else:
        return html.Div(['Data: ', html.A('select')])

@app.callback(
    Output('import_data2', 'children'),
    [
    Input('import_data2', 'contents'),
    Input('import_data2', 'filename'),
    ]
)
def import_data2(contents, filename):
    if contents is not None:
        contents = contents
        filename = filename
        global df2
        df2 = parse_data(contents, filename)
        return html.Div([html.A(filename)])
    else:
        return html.Div(['Data: ', html.A('select')])


@app.callback(
    Output('import_data3', 'children'),
    [
    Input('import_data3', 'contents'),
    Input('import_data3', 'filename'),
    ]
)
def import_data3(contents, filename):
    if contents is not None:
        contents = contents
        filename = filename
        global df3
        df3 = parse_data(contents, filename)
        return html.Div([html.A(filename)])
    else:
        return html.Div(['Data: ', html.A('select')])


if __name__ == "__main__":
    app.run_server(debug=False)