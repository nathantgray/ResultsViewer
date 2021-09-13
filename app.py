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
                    margin={'t': 15, 'l': 5, 'b': 5, 'r': 5},
                    # plot_bgcolor=colors["graphBackground"],
                    # paper_bgcolor=colors["graphBackground"]
                ))
global df1
global df2
global df3
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

app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                [
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
                ],
                width={'size': 5, 'offset': 1}
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
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
                        cyto.Cytoscape(
                            id="cytoscape",
                            elements=diagram.elements,
                            # stylesheet=stylesheet,
                            layout={"name": "preset", "fit": True, "animate": True},
                            style={
                                "height": "650px",
                                "width": "100%",
                                "backgroundColor": "white",
                                "margin": "auto",
                            },
                            minZoom=0.3,
                        ),
                    ],
                    width={'size': 5, 'offset': 1}
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Card(
                                    [
                                        dcc.Upload(
                                            id='import_data1',
                                            children=html.Div(['Select Node Data for Graph 1: ', html.A('file')]),
                                            multiple=False
                                        ),
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
                                        dcc.Upload(
                                            id='import_data2',
                                            children=html.Div(['Select Node Data for Graph 2: ', html.A('file')]),
                                            multiple=False
                                        ),
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
                                        dcc.Upload(
                                            id='import_data3',
                                            children=html.Div(['Select Node Data for Graph 3: ', html.A('file')]),
                                            multiple=False
                                        ),
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
    Input("cytoscape", "selectedNodeData")
    ]
)
def update_graph1(nodedata):
    if nodedata is not None and df1 is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = df1
                y = np.array(df[f'node_{node_name}'])
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
                    margin={'t': 15, 'l': 5, 'b': 5, 'r': 5},
                    # plot_bgcolor=colors["graphBackground"],
                    # paper_bgcolor=colors["graphBackground"],
                    showlegend=True
                )
            )
    return figure


@app.callback(
    Output('node_graph2', 'figure'), [
    Input("cytoscape", "selectedNodeData")
    ]
)
def update_graph2(nodedata):
    if nodedata is not None and df2 is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = df2
                y = np.array(df[f'node_{node_name}'])
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
                    margin={'t': 15, 'l': 5, 'b': 5, 'r': 5},
                    # plot_bgcolor=colors["graphBackground"],
                    # paper_bgcolor=colors["graphBackground"],
                    showlegend=True
                )
            )
    return figure


@app.callback(
    Output('node_graph3', 'figure'), [
    Input("cytoscape", "selectedNodeData")
    ]
)
def update_graph3(nodedata):
    if nodedata is not None and df3 is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = df3
                y = np.array(df[f'node_{node_name}'])
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
                    margin={'t': 15, 'l': 5, 'b': 5, 'r': 5},
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
        return html.Div(['Select Node Data: ', html.A(filename)])
    else:
        return html.Div(['Select Node Data: ', html.A('file')])

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
        return html.Div(['Select Node Data: ', html.A(filename)])
    else:
        return html.Div(['Select Node Data: ', html.A('file')])


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
        return html.Div(['Select Node Data: ', html.A(filename)])
    else:
        return html.Div(['Select Node Data: ', html.A('file')])


if __name__ == "__main__":
    app.run_server(debug=False)