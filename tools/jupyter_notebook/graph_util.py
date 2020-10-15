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

def draw_two_graph(df1, df2, y, x, color, figsize, legend, title, xlabel, ylabel, start_layer, end_layer):
    df1.columns = df1.columns.str.strip()
    if(end_layer == "all"):
        df1 = df1
    else:
        df1 = df1[start_layer:end_layer]
        
    df2.columns = df2.columns.str.strip()
    if(end_layer == "all"):
        df2 = df2
    else:
        df2 = df2[start_layer:end_layer]
    
    fig, axes = plt.subplots(nrows=1, ncols=1)

    df1.plot(kind='Line', y = y, x = x, color = 'red', figsize = figsize, legend = legend, ax=axes)
    df2.plot(kind='Line', y = y, x = x, color = 'blue', figsize = figsize, legend = legend, ax=axes)
        
    #my_plot = df_new.plot(kind='bar', y = y, x = x, color = color, figsize = figsize, legend = legend)
    #fig.set_title(title)
    #fig.set_xlabel(xlabel)
    #fig.set_ylabel(ylabel)
    return fig