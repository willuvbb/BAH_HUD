

import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

df = pd.read_csv('./../AllTweets.csv')

app = dash.Dash(__name__)

app.layout = html.Div([
    dash_table.DataTable(id='table', sorting=True,
          columns=[{"name": i, "id": i} for i in df.columns],
          style_cell={'maxWidth': '400px', 'whiteSpace': 'normal'},
          data=df.to_dict())
])



if __name__ == '__main__':
    app.run_server(debug=True,port=8061)