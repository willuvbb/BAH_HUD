import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import pandas as pd

tweets = pd.read_csv('./../AllTweets.csv')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Markdown('''
        #### An Example Title Using Dash and Markdown
        
        Dash supports [Markdown](http://commonmark.org/help).
        
        Markdown is a simple way to write and format text.
        It includes a syntax for things like **bold text** and *italics*,
        [links](http://commonmark.org/help), inline `code` snippets, lists,
        quotes, and more.
        '''),

        html.Div(id='container_prim_filter',
                 children=dcc.Dropdown(id='prim_filter', multi=True,
                                       options=[{'label': cat, 'value': cat}
                                                for cat in tweets['Primary'].unique()]
                                       )),

        html.Div(id='container_prim_checklist',
                 children=dcc.Checklist(id='prim_checklist',
                                        options=[
                                            {'label': cat, 'value': cat} for cat in tweets['Primary'].unique()],
                                        value=[cat for cat in tweets['Primary'].unique()],
                                        labelStyle={'display': 'inline-block'}
                                        )),
    ]),

    html.Div([
        dcc.Markdown('''
        ### Below, we have our data table!
        '''),

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
            columns=[{'id': c, 'name': c} for c in tweets.columns]
        )
    ]),
])


@app.callback(Output('table', 'data'),
              [Input('prim_checklist', 'value')])
def filter_table(primary):
    if all([param is None for param in [primary]]):
        raise PreventUpdate
    dff = tweets[tweets['Primary'].isin(primary)]
    return dff.to_dict('records')


# @app.callback(Output('table', 'data'),
#               [Input('prim_filter', 'value')])
# def filter_table(primary):
#     if all([param is None for param in [primary]]):
#         raise PreventUpdate
#     dff = tweets[tweets['Primary'].isin(primary)]
#     return tweetsf.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True,port=8063)