from pathlib import Path
import plotly.express as px
from glmpy import Gridlabd as gld
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_outputs(outputs):
    v_a = gld.read_csv(outputs / "nodes_volts_A_mag.csv", parse_dates=True)
    v_b = gld.read_csv(outputs / "nodes_volts_B_mag.csv", parse_dates=True)
    v_c = gld.read_csv(outputs / "nodes_volts_C_mag.csv", parse_dates=True)
    fig = make_subplots(rows=3, cols=2, start_cell="top-left")

    fig.add_trace(go.Scatter(x=v_a.index, y=v_a["1212526"]),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=v_b.index, y=v_b["1212526"]),
                  row=2, col=2)

    fig.add_trace(go.Scatter(x=v_c.index, y=v_c["1212526"]),
                  row=3, col=2)

    # fig.add_trace(go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000]),
    #               row=2, col=2)


    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["y", v_a["1212526"]],
                        label="1212526",
                        method="restyle"
                    ),
                    dict(
                        args=["y", "heatmap"],
                        label="Heatmap",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ], row=1, col=2
    )

    fig.show()
if __name__ == '__main__':
    out_dir = Path("/home/nathangray/PycharmProjects/EPBCosim/gridlabd/EPB_CA/output")
    plot_outputs(out_dir)
