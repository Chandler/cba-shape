import chart_studio.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.io as pio

def basic_histogram(x):
    fig = go.Figure(data=[go.Histogram(x=x)])
    pio.write_html(fig, file="/tmp/plotly.html", auto_open=True)


def basic_heatmap(z):
    fig = go.Figure(data=[go.Heatmap(z=z)])
    pio.write_html(fig, file="/tmp/plotly.html", auto_open=True)
