import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

layout = html.Div([
    html.Div([
        html.Button(
            'Show Dashboard',
            id='show-dash-button',
            style={
                'top': '15%',
                'left': '43.5%',
                'float': 'center',
                'position': 'fixed',
            }
        ),
        html.Img(
            src='/assets/HUD_Seal.svg',
            style={
                # 'height': '50%',
                # 'width': '50%',
                'top': '50%',
                'left': '50%',
                'float': 'center',
                'position': 'fixed',
                'margin-top': -182,
                'margin-left': -182
            }
        ),
    ],)
    # dcc.Link('Go to Dashboard', href='/apps/dashboard')
])

@app.callback(
    Output('url', 'pathname'),
    [Input('show-dash-button', 'n_clicks')]
)
def button_click(n_clicks):
    # When you click the button, change the end of the url to 'ABC'
    return '/apps/dashboard'

# @app.callback(
#     Output('app-2-display-value', 'children'),
#     [Input('app-2-dropdown', 'value')])
# def display_value(value):
#     return 'You have selected "{}"'.format(value)