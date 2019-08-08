import dash
import dash_bootstrap_components as dbc

external_stylesheets = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# external_stylesheets = ['MyStyle.css']
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
my_custom_style = './assets/custom-styles.css'
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID, external_stylesheets, my_custom_style])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, external_stylesheets])
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True