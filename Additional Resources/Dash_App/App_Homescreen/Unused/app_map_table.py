import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table

from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

tweets = pd.read_csv('AllTweets_WithStates.csv')

states = pd.read_csv('StatesWithCounts.csv')

# Format the state data
for col in states.columns:
    states[col] = states[col].astype(str)

states['text'] = states['Abbreviation'] + '<br>' + \
    states['count'] + ' Tweets'


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    html.Div([
            dcc.Markdown('''
            #### HUD Twitter Dashboard
            '''),

            html.Img(src='/assets/HUD_Seal.svg',
                     style={
                         'height': '4%',
                         'width': '4%',
                         'float': 'center',
                         'position': 'relative',
                         'padding-top': 0,
                         'padding-right': 0
                     }
                     )
        ]),

    dcc.Tabs(id="tabs-styled-with-inline", value='tab-data-table', children=[
        dcc.Tab(label='Data Table', value='tab-data-table', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Map', value='tab-map', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline')
])

@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])
def render_content(tab):
    if tab == 'tab-data-table':
        return html.Div([
            html.H3('Data Table Here'),

            dash_table.DataTable(
                id='table',
                style_data={'whiteSpace': 'normal'},
                css=[{
                    'selector': '.dash-cell div.dash-cell-value',
                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                }],
                # fixed_rows={ 'headers': True, 'data': 0 },
                sort_action="native",
                filter_action='native',
                style_cell={'textAlign': 'center'},
                # style_cell_conditional=[
                #     {
                #         'if': {'column_id': 'Region'},
                #         'textAlign': 'left'
                #     }
                # ],
                data=tweets.to_dict('records'),
                columns=[{'id': c, 'name': c.replace('_', ' ').title()} for c in tweets.columns]
            )
        ])
    elif tab == 'tab-map':
        return html.Div([
            html.H3('Heat Map'),

            dcc.Graph(
                figure=go.Figure(
                    data=go.Choropleth(
                        locations=states['Abbreviation'],
                        z=states['count'].astype(float),
                        locationmode='USA-states',
                        colorscale='Blues',
                        autocolorscale=False,
                        text=states['text'],  # hover text
                        marker_line_color='white',  # line markers between states
                        colorbar_title="Number of Tweets"
                    ),
                    layout=go.Layout(
                        title_text='Public Housing Tweets by State<br>(Hover for breakdown)',
                        geo=dict(
                            scope='usa',
                            projection=go.layout.geo.Projection(type='albers usa'),
                            showlakes=True,  # lakes
                            lakecolor='rgb(255, 255, 255)'),
                    )
                )
            )
        ])

if __name__ == '__main__':
    app.run_server(debug=True,port=8066)