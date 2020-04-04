import chart_studio.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.io as pio


def plot_line_segments(x, y, z):
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        marker=dict(size=4, color=z, colorscale="Viridis",),
        line=dict(color="#1f77b4", width=1),
    )

    data = [trace]

    layout = dict(
        width=1000,
        height=1000,
        autosize=False,
        scene=dict(
            xaxis=dict(
                gridcolor="rgb(255, 255, 255)",
                zerolinecolor="rgb(255, 255, 255)",
                showbackground=True,
                backgroundcolor="rgb(230, 230,230)",
            ),
            yaxis=dict(
                gridcolor="rgb(255, 255, 255)",
                zerolinecolor="rgb(255, 255, 255)",
                showbackground=True,
                backgroundcolor="rgb(230, 230,230)",
            ),
            zaxis=dict(
                gridcolor="rgb(255, 255, 255)",
                zerolinecolor="rgb(255, 255, 255)",
                showbackground=True,
                backgroundcolor="rgb(230, 230,230)",
            ),
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1, y=1, z=1,)),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="manual",
        ),
    )

    fig = dict(data=data, layout=layout)

    pio.write_html(fig, file="/tmp/plotly.html", auto_open=True)


def basic_histogram(x):
    fig = go.Figure(data=[go.Histogram(x=x)])
    pio.write_html(fig, file="/tmp/plotly.html", auto_open=True)


def basic_heatmap(z):
    fig = go.Figure(data=[go.Heatmap(z=z)])
    pio.write_html(fig, file="/tmp/plotly.html", auto_open=True)
