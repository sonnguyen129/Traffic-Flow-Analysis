from datetime import date
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR, dbc.icons.BOOTSTRAP,'https://fonts.googleapis.com/css2?family=Montserrat'])

inputs = dbc.Form([
    ## hide these 2 inputs if file is uploaded

    ## always visible
    dbc.Label("Number of Trials", html_for="n-iter"), 
    html.P(id='title', style={"marginLeft":"20px"}),
    dcc.Slider(id="n-iter", min=10, max=1000, step=None, marks={10:"10", 100:"100", 500:"500", 1000:"1000"}, value=0),

    ## run button
    html.Br(),html.Br(),
    dbc.Col(html.A(dbc.Button("run", id="run", color="primary"))),
    
])


@app.callback(output=[Output(component_id="title", component_property="children")],
              inputs=[Input(component_id="run", component_property="n_clicks")],
              state=[State("n-iter","value")])
def results(n_clicks, n_iter):
    global app;
    print(n_iter)
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP,'https://fonts.googleapis.com/css2?family=Montserrat'])
    return [str(n_iter)]
    


app.layout = dbc.Container(fluid=True, children=[
    html.H1("ghjghjghjghggjgjg", id="nav-pills"),
    inputs
])

########################## Run ##########################
if __name__ == "__main__":
    app.run_server(debug=True)