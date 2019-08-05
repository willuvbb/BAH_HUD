import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go

tweets = pd.read_csv('./../AllTweets.csv')

states = pd.read_csv('./../StatesWithCounts.csv')

# Format the state data
for col in states.columns:
    states[col] = states[col].astype(str)

states['text'] = states['Abbreviation'] + '<br>' + \
    states['count'] + ' Tweets'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        dcc.Markdown('''
        #### HUD Twitter Dashboard
        '''),
    ]),

    html.Div([
        dcc.Markdown('''
        ### Look, a heat map!
        '''),

        dcc.Graph(
            figure=go.Figure(
                data=go.Choropleth(
                        locations=states['Abbreviation'],
                        z=states['count'].astype(float),
                        locationmode='USA-states',
                        colorscale='Blues',
                        autocolorscale=False,
                        text=states['text'], # hover text
                        marker_line_color='white', # line markers between states
                        colorbar_title="Number of Tweets"
                ),
                layout=go.Layout(
                    title_text='Public Housing Tweets by State<br>(Hover for breakdown)',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                )
            )
        )
    ]),
])



if __name__ == '__main__':
    app.run_server(debug=True, port=8063)