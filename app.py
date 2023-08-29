import dash

metaTags = [
    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5'}]

external_stylesheets = [metaTags]

app = dash.Dash(__name__, meta_tags=metaTags, external_stylesheets = external_stylesheets,title = 'Python4Trading',
                suppress_callback_exceptions=True)

server = app.server
