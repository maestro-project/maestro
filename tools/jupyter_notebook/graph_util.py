import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_graph(df, y, x, color, figsize, legend, title, xlabel, ylabel, start_layer, end_layer):
    df.columns = df.columns.str.strip()
    if(end_layer == "all"):
        df_new = df
    else:
        df_new = df[start_layer:end_layer]
    my_plot = df_new.plot(kind='bar', y = y, x = x, color = color, figsize = figsize, legend = legend)
    my_plot.set_title(title)
    my_plot.set_xlabel(xlabel)
    my_plot.set_ylabel(ylabel)
    return my_plot
