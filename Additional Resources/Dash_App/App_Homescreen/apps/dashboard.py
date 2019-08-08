import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
import plotly.graph_objs as go
from app import app
from textwrap import dedent as d
import json

tweets = pd.read_csv('AllTweets_WithStates.csv')

states = pd.read_csv('StatesWithCounts.csv')

# content_count = tweets['content'].value_counts()
# emotion_count = tweets['emotion'].value_counts()

# Format the state data
for col in states.columns:
    states[col] = states[col].astype(str)

states['text'] = states['Abbreviation'] + '<br>' + \
    states['count'] + ' Tweets'

tabs_styles = {
    'height': '44px',
    'display': 'inline-block'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'width': 'calc(100%/3)',
    'display': 'inline-block'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'width': 'calc(100%/3)',
    'display': 'inline-block'
}

filter_data = None

layout = html.Div([
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Markdown(
                                children='''
                                    #### HUD Twitter Dashboard
                                    ''',
                                style={
                                    'textAlign': 'center'
                                }
                            )
                        ],
                        md=12,
                        align="center"
                    )
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id='filter_data', children=None, style={'display': 'none'}),

                            dcc.Tabs(id="tabs-styled-with-inline",
                                     value='tab-data-table',
                                     children=[
                                         dcc.Tab(label='Data Table', value='tab-data-table', style=tab_style,
                                                 selected_style=tab_selected_style, className='mobileTab'),
                                         dcc.Tab(label='Heat Map', value='tab-map', style=tab_style,
                                                 selected_style=tab_selected_style, className='mobileTab'),
                                         dcc.Tab(label='Charts', value='tab-charts', style=tab_style,
                                                 selected_style=tab_selected_style, className='mobileTab'),
                                     ],
                                     style=tabs_styles),
                        ]
                    )
                ]
            ),
    ],
    fluid=True
    ),

    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id='tabs-content-inline')
                        ]
                    )
                ]
            )
        ],
    fluid=True
    ),
    # html.Div([
    #     dcc.Markdown('''
    #     #### HUD Twitter Dashboard
    #     '''),
    # ]),
    dcc.Link('Go to Home', href='/')
])

# restyleData
@app.callback(
    Output('relayout-data-display', 'children'),
    [Input('prim_chart', 'relayoutData')])
def display_selected_data(relayout_data):
    return json.dumps(relayout_data, indent=2)

@app.callback(
    Output('tabs-content-inline', 'children'),
    # [Input('tabs-styled-with-inline', 'value'),
    #  Input('prim_chart', 'relayoutData')]
    [Input('tabs-styled-with-inline', 'value')]
)
def render_content(tab):
    global filter_data
    if tab == 'tab-data-table':
        if filter_data is not None:
            my_query = '{state} contains ' + filter_data
            print('my_query: ' + my_query)
            return html.Div([
                dash_table.DataTable(
                    id='table',
                    style_data={'whiteSpace': 'normal'},
                    css=[{
                        'selector': '.dash-cell div.dash-cell-value',
                        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                    }],
                    # fixed_rows={'headers': True},
                    sort_action="native",
                    filter_action='native',
                    filter_query=my_query,
                    style_cell={'textAlign': 'center', 'fontSize': 16, 'font-family': 'sans-serif',
                                'maxWidth': '150px'},
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'followers'},
                            'textAlign': 'right',
                            'width': '50px'
                        },
                        {
                            'if': {'column_id': 'tweet'},
                            'textAlign': 'center',
                            'width': '500px'
                        },
                        {
                            'if': {'column_id': 'location'},
                            'textAlign': 'center',
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'username'},
                            'textAlign': 'center',
                            'width': '100px'
                        }
                    ],
                    data=tweets.to_dict('records'),
                    columns=[{'id': c, 'name': c.replace('_', ' ').title()} for c in tweets.columns]
                )
            ])
        else:
            return html.Div([
                dash_table.DataTable(
                    id='table',
                    style_data={'whiteSpace': 'normal'},
                    fill_width=True,
                    css=[{
                        'selector': '.dash-cell div.dash-cell-value',
                        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                    }],
                    # fixed_rows={'headers': True},
                    sort_action="native",
                    filter_action='native',
                    # style_cell={'textAlign': 'center', 'fontSize': 16, 'font-family': 'sans-serif',
                    #             'maxWidth': '150px'},
                    style_cell={'textAlign': 'center', 'fontSize': 16, 'font-family': "Helvetica Neue",
                                'maxWidth': '150px'},


                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'followers'},
                            'textAlign': 'right',
                            'width': '50px'
                        },
                        {
                            'if': {'column_id': 'tweet'},
                            'textAlign': 'center',
                            'width': '500px'
                        },
                        {
                            'if': {'column_id': 'location'},
                            'textAlign': 'center',
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'username'},
                            'textAlign': 'center',
                            'width': '100px'
                        }
                    ],
                    data=tweets.to_dict('records'),
                    columns=[{'id': c, 'name': c.replace('_', ' ').title()} for c in tweets.columns]
                )
            ])
    elif tab == 'tab-map':
        filter_data = None
        return html.Div([
            html.H3('Heat Map'),

            dcc.Graph(
                id='map',
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
                        width=1200,
                        height=600,
                        geo=dict(
                            scope='usa',
                            projection=go.layout.geo.Projection(type='albers usa'),
                            showlakes=True,  # lakes
                            lakecolor='rgb(255, 255, 255)'),
                    )
                )
            )
        ])
    elif tab == 'tab-charts':
        filter_data = None
        return html.Div([

            dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # html.H3('Chart 1'),

                                dcc.Graph(
                                    id='prim_chart',
                                    figure=go.Figure(
                                        data=go.Pie(
                                            values=tweets['content'].value_counts().values,
                                            labels=tweets['content'].value_counts().index,
                                            textinfo='label',
                                            hole=.3
                                        ),
                                        layout=go.Layout(
                                            title_text='Primary Category Breakdown (Hover for count)',
                                            # autosize=True
                                            # width=1200,
                                            # height=600
                                        )
                                    )
                                )
                            ],
                            lg=6,
                        ),
                        dbc.Col([
                            dcc.Graph(
                                id='sec_chart',
                                figure=go.Figure(
                                    data=go.Pie(

                                        values=tweets['emotion'].value_counts().values,
                                        labels=tweets['emotion'].value_counts().index,
                                        textinfo='label',
                                        hole=.3
                                    ),
                                    layout=go.Layout(
                                        title_text='Secondary Category Breakdown (Hover for count)',
                                    )
                                )
                            )],lg=6,
                        ),
                    ]
                ),

                dbc.Row(
                    [
                        html.Div([
                            dcc.Markdown(d("""
                                **Restyle Data**

                                Clicking on legend items will fire
                                this event.
                            """)),
                            html.Pre(
                                id='relayout-data-display',
                                children='not restyled yet'),
                        ], className='three columns')
                    ]
                )
            ],
                # className="mt-4",
                fluid=True
            ),
        ])
# def render_content(tab, hiddenlabels):
#     global filter_data
#     if tab == 'tab-data-table':
#         if filter_data is not None:
#             my_query = '{state} contains ' + filter_data
#             print('my_query: ' + my_query)
#             return html.Div([
#                 dash_table.DataTable(
#                     id='table',
#                     style_data={'whiteSpace': 'normal'},
#                     css=[{
#                         'selector': '.dash-cell div.dash-cell-value',
#                         'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
#                     }],
#                     # fixed_rows={'headers': True},
#                     sort_action="native",
#                     filter_action='native',
#                     filter_query=my_query,
#                     style_cell={'textAlign': 'center', 'fontSize': 16, 'font-family': 'sans-serif',
#                                 'maxWidth': '150px'},
#                     style_cell_conditional=[
#                         {
#                             'if': {'column_id': 'followers'},
#                             'textAlign': 'right',
#                             'width': '50px'
#                         },
#                         {
#                             'if': {'column_id': 'tweet'},
#                             'textAlign': 'center',
#                             'width': '500px'
#                         },
#                         {
#                             'if': {'column_id': 'location'},
#                             'textAlign': 'center',
#                             'width': '100px'
#                         },
#                         {
#                             'if': {'column_id': 'username'},
#                             'textAlign': 'center',
#                             'width': '100px'
#                         }
#                     ],
#                     data=tweets.to_dict('records'),
#                     columns=[{'id': c, 'name': c.replace('_', ' ').title()} for c in tweets.columns]
#                 )
#             ])
#         else:
#             return html.Div([
#                 dash_table.DataTable(
#                     id='table',
#                     style_data={'whiteSpace': 'normal'},
#                     fill_width=True,
#                     css=[{
#                         'selector': '.dash-cell div.dash-cell-value',
#                         'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
#                     }],
#                     # fixed_rows={'headers': True},
#                     sort_action="native",
#                     filter_action='native',
#                     # style_cell={'textAlign': 'center', 'fontSize': 16, 'font-family': 'sans-serif',
#                     #             'maxWidth': '150px'},
#                     style_cell={'textAlign': 'center', 'fontSize': 16, 'font-family': "Helvetica Neue",
#                                 'maxWidth': '150px'},
#
#
#                     style_cell_conditional=[
#                         {
#                             'if': {'column_id': 'followers'},
#                             'textAlign': 'right',
#                             'width': '50px'
#                         },
#                         {
#                             'if': {'column_id': 'tweet'},
#                             'textAlign': 'center',
#                             'width': '500px'
#                         },
#                         {
#                             'if': {'column_id': 'location'},
#                             'textAlign': 'center',
#                             'width': '100px'
#                         },
#                         {
#                             'if': {'column_id': 'username'},
#                             'textAlign': 'center',
#                             'width': '100px'
#                         }
#                     ],
#                     data=tweets.to_dict('records'),
#                     columns=[{'id': c, 'name': c.replace('_', ' ').title()} for c in tweets.columns]
#                 )
#             ])
#     elif tab == 'tab-map':
#         filter_data = None
#         return html.Div([
#             html.H3('Heat Map'),
#
#             dcc.Graph(
#                 id='map',
#                 figure=go.Figure(
#                     data=go.Choropleth(
#                         locations=states['Abbreviation'],
#                         z=states['count'].astype(float),
#                         locationmode='USA-states',
#                         colorscale='Blues',
#                         autocolorscale=False,
#                         text=states['text'],  # hover text
#                         marker_line_color='white',  # line markers between states
#                         colorbar_title="Number of Tweets"
#                     ),
#                     layout=go.Layout(
#                         title_text='Public Housing Tweets by State<br>(Hover for breakdown)',
#                         width=1200,
#                         height=600,
#                         geo=dict(
#                             scope='usa',
#                             projection=go.layout.geo.Projection(type='albers usa'),
#                             showlakes=True,  # lakes
#                             lakecolor='rgb(255, 255, 255)'),
#                     )
#                 )
#             )
#         ])
#     elif tab == 'tab-charts':
#         filter_data = None
#         print(type(hiddenlabels))
#         return html.Div([
#
#             dbc.Container(
#             [
#                 dbc.Row(
#                     [
#                         dbc.Col(
#                             [
#                                 # html.H3('Chart 1'),
#
#                                 dcc.Graph(
#                                     id='prim_chart',
#                                     figure=go.Figure(
#                                         data=go.Pie(
#                                             values=tweets['content'].value_counts().values,
#                                             labels=tweets['content'].value_counts().index,
#                                             textinfo='label',
#                                             hole=.3
#                                         ),
#                                         layout=go.Layout(
#                                             title_text='Primary Category Breakdown (Hover for count)',
#                                             # autosize=True
#                                             # width=1200,
#                                             # height=600
#                                         )
#                                     )
#                                 )
#                             ],
#                             lg=6,
#                         ),
#                         dbc.Col([
#                             dcc.Graph(
#                                 id='sec_chart',
#                                 figure=go.Figure(
#                                     data=go.Pie(
#                                         # tweets[~tweets.content.isin(hiddenlabels)]
#
#                                         values=tweets['emotion'].value_counts().values,
#                                         labels=tweets['emotion'].value_counts().index,
#                                         textinfo='label',
#                                         hole=.3
#                                     ),
#                                     layout=go.Layout(
#                                         title_text='Secondary Category Breakdown (Hover for count)',
#                                     )
#                                 )
#                             )],lg=6,
#                         ),
#                     ]
#                 ),
#
#                 dbc.Row(
#                     [
#                         html.Div([
#                             dcc.Markdown(d("""
#                                 **Restyle Data**
#
#                                 Clicking on legend items will fire
#                                 this event.
#                             """)),
#                             html.Pre(
#                                 id='relayout-data-display',
#                                 children='not restyled yet'),
#                         ], className='three columns')
#                     ]
#                 )
#             ],
#                 # className="mt-4",
#                 fluid=True
#             ),
#         ])

# @app.callback(
#     [Output('filter_data', 'value'),
#      Output('tabs-styled-with-inline', 'value')],
#     [Input('clear_filter_button', 'n_clicks')])
# def clear_filter(cf_button_clicks):
#     if cf_button_clicks is not 0:
#         return None, 'tab-data-table'

# @app.callback(
#     Output('test_text_2','children'),
#     [Input('clear_filter_button','n_clicks')]
# )
# def print_n_clicks(n_clicks):
#     return str('the clear filter button has been pressed {} times',n_clicks)


# @app.callback(
#     [Output('filter_data', 'value'),
#      Output('tabs-styled-with-inline', 'value')],
#     [Input('clear_filter_button', 'n_clicks'),
#      Input('map', 'clickData')])
# def display_selected_state(cf_button_clicks, selection):
#     print('cf button clicks:')
#     print(cf_button_clicks)
#     if cf_button_clicks is None:
#         if selection is None:
#             return {}
#         else:
#             # return dcc.Markdown('''You clicked something!''')
#             # return dcc.Markdown(str(selection))
#             print('variable: "selection" within the function')
#             print(selection)
#             print(type(selection))
#             return selection, 'tab-data-table'
#     else:
#         return None, 'tab-data-table'

@app.callback(
    [Output('filter_data', 'children'),
     Output('tabs-styled-with-inline', 'value')],
    [Input('map', 'clickData')])
def display_selected_state(selection):
        if selection is None:
            return {}
        else:
            # return dcc.Markdown('''You clicked something!''')
            # return dcc.Markdown(str(selection))
            print('variable: "selection" within the function')
            print(selection)
            print(type(selection))
            global filter_data
            filter_data = selection['points'][0]['location']
            return selection['points'][0]['location'], 'tab-data-table'