import math
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import dash_html_components as html
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


# global tree
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    # external_stylesheets=dbc.themes.BOOTSTRAP
)
server = app.server

colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}
PATH = Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# Define elements, stylesheet and layout
diagram = Diagram(Path('./data/ieee123.drawio'))

figure = go.Figure(
                data=[
                    go.Scatter(
                        x=np.array(range(0, 10)),
                        y=np.array(range(0, 10)),
                        mode='lines+markers')
                    ],
                layout=go.Layout(
                    plot_bgcolor=colors["graphBackground"],
                    paper_bgcolor=colors["graphBackground"]
                ))
global volt_df
stylesheet = [
    {
        "selector": ".nonterminal",
        "style": {
            "label": "data(confidence)",
            "background-opacity": 0,
            "text-halign": "left",
            "text-valign": "top",
        },
    },
    {"selector": ".support", "style": {"background-opacity": 0}},
    {
        "selector": "edge",
        "style": {
            "source-endpoint": "inside-to-node",
            "target-endpoint": "inside-to-node",
        },
    },
    {
        "selector": ".terminal",
        "style": {
            "label": "data(name)",
            "width": 10,
            "height": 10,
            "text-valign": "center",
            "text-halign": "right",
            "background-color": "#222222",
        },
    },
]

app.layout = html.Div(
    [
        html.Img(className="logo", src=app.get_asset_url("dash-logo.png")),
        html.Div(
            className="header",
            children=[
                html.Div(
                    className="div-info",
                    children=[
                        html.H2(className="title", children="GridLAB-D Interface"),
                        html.P(
                            """
                            Dash Cytoscape is a graph visualization component for creating easily customizable,
                            high-performance interactive, and web-based networks.
                            """
                        ),
                        html.A(
                            children=html.Button("Run Model", id='run_model', className="button"),
                            # href="https://www.gridlabd.org/index.stm",
                            target="_blank",
                        ),
                        dcc.Upload(
                            id='import_diagram',
                            children=html.Div(['Select Diagram: ', html.A('diagram')]),
                            style={
                                    'width': '25%',
                                    'height': '25px',
                                    'lineHeight': '30px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center'
                                    },
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
                        dcc.Upload(
                            id='import_data',
                            children=html.Div(['Select Node Data: ', html.A('file')]),
                            style={
                                    'width': '25%',
                                    'height': '25px',
                                    'lineHeight': '30px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center'
                                    },
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
                    ],
                ),
                html.H4("Network"),
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
                    minZoom=0.45,
                ),
                dbc.Row(
                    [
                        dbc.Alert(
                            id="node-data",
                            children="Click on a node to see its details here",
                            color="secondary",
                        ),
                        dcc.Graph(
                            id="node_graph",
                            figure=go.Figure(
                                data=[
                                    go.Scatter(
                                        x=np.array(range(0, 10)),
                                        y=np.array(range(0, 10)),
                                        mode='lines+markers')
                                ],
                                layout=go.Layout(
                                    plot_bgcolor=colors["graphBackground"],
                                    paper_bgcolor=colors["graphBackground"]
                                ))
                        ),
                        dcc.Graph(
                            id="edge_graph",
                            figure=go.Figure(
                                data=[
                                    go.Scatter(
                                        x=np.array(range(0, 10)),
                                        y=np.array(range(0, 10)),
                                        mode='lines+markers')
                                ],
                                layout=go.Layout(
                                    plot_bgcolor=colors["graphBackground"],
                                    paper_bgcolor=colors["graphBackground"]
                                ))
                        )
                    ]
                ),
            ],
        ),
    ]
)


# @app.callback(
#     Output("cytoscape", "stylesheet"), [Input("cytoscape", "mouseoverEdgeData")]
# )
# def color_children(edgeData):
#     if edgeData is None:
#         return stylesheet
#
#     if "s" in edgeData["source"]:
#         val = edgeData["source"].split("s")[0]
#     else:
#         val = edgeData["source"]
#
#     children_style = [
#         {"selector": f'edge[source *= "{val}"]', "style": {"line-color": "#3ed6d2"}}
#     ]
#
#     return stylesheet + children_style

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
    Output('node_graph', 'figure'), [
    Input("cytoscape", "selectedNodeData")
    ]
)
def update_graph(nodedata):
    if nodedata is not None and volt_df is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = volt_df
                y=np.array(df[f'node_{node_name}'])
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
                    plot_bgcolor=colors["graphBackground"],
                    paper_bgcolor=colors["graphBackground"],
                    showlegend=True
                )
            )
    return figure

@app.callback(
    Output('edge_graph', 'figure'), [
    Input("cytoscape", "selectedEdgeData")
    ]
)
def update_graph(nodedata):
    if nodedata is not None and volt_df is not None:
        if len(nodedata) > 0:
            fig_data = []
            for data in nodedata:
                node_name = data["label"]
                df = volt_df
                y=np.array(df[f'node_{node_name}'])
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
                    plot_bgcolor=colors["graphBackground"],
                    paper_bgcolor=colors["graphBackground"],
                    showlegend=True
                )
            )
    return figure

@app.callback(
    Output('import_data', 'children'),
    [
    Input('import_data', 'contents'),
    Input('import_data', 'filename'),
    ]
)
def import_data(contents, filename):
    if contents is not None:
        contents = contents
        filename = filename
        global volt_df
        volt_df = parse_data(contents, filename)
        return html.Div(['Select Node Data: ', html.A(filename)])
    else:
        return html.Div(['Select Node Data: ', html.A('file')])



if __name__ == "__main__":
    app.run_server(debug=False)
