#! /home/nathangray/PycharmProjects/ResultsViewer/venv310/bin/python3.10
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import io
import base64
from math import sqrt


app = Dash(__name__)
df = px.data.stocks()
base = sqrt(3)
# df = pd.DataFrame  # replace with your own data source
options = list(df.iloc[0, 1:].keys())
print(options)
app.layout = html.Div([
    html.H4('Dataframe plotter'),
    dcc.Upload(id='uploader',
               children=html.Div(['Select csv: ', html.A('file')]),
               multiple=False),
    dcc.Graph(id="time-series-chart"),
    html.P("Select column:"),
    dcc.Input(id="base_input", placeholder="Base", type="number", persistence=True, persistence_type='local'),
    dcc.Dropdown(
        id="ticker",
        options=options,
        # value=options[0],
        clearable=False,
        multi=True
    ),
])


@app.callback(
    [
        Output('ticker', 'options'),
        Output('uploader', 'children'),
    ],
    [
    Input('uploader', 'contents'),
    Input('uploader', 'filename'),
    ]
)
def import_data1(contents, filename):
    if contents is not None:
        contents = contents
        filename = filename
        global df, options
        df = parse_data(contents, filename)
        options = list(df.iloc[0, 1:].keys())
        return options, html.Div([html.A(filename)])
    else:
        return options, html.Div(['Data: ', html.A('select')])


@app.callback(
    Output("time-series-chart", "figure"),
    [
        Input("ticker", "value"),
        Input("base_input", "value")
    ]
)
def display_time_series(ticker, base):
    print(ticker)
    _df = df.copy()
    _df[ticker] = df[ticker]/(base/sqrt(3))
    fig = px.line(_df, y=ticker)
    return fig

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
if __name__ == '__main__':
    app.run_server(debug=False)