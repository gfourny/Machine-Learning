import plotly
plotly.tools.set_credentials_file(username='gfourny', api_key='Thm6FKxLGWxTMDZ0VKCT')
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

import numpy as np
import pandas as pd

df = pd.read_csv('master.csv')
master_data_table = FF.create_table(df.head())
py.iplot(master_data_table, filename='master_data_table')

trace1 = go.Scatter(
                    x=df['country'], y=df['suicides_no'], # Data
                    mode='lines', name='nombre de suicide' # Additional options
)

layout = go.Layout(title='Suicide par pays',
                   plot_bgcolor='rgb(230, 230,230)')

fig = go.Figure(data=[trace1], layout=layout)

# Plot data in the notebook
py.iplot(fig, filename='simple-plot-from-csv')