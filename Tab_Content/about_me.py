import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from app import app


about_me_LO = html.Div([


    html.Div([

        html.H4("About me:"),

        html.Div([

            html.P("I am Sathya Jayagopi, a seasoned Software Developer with over 12 years of experience spanning Software Development and Test Management. My passion lies in crafting Interactive Data Analytic web applications, where I take immense pleasure in uncovering hidden patterns within intricate data sets, revealing actionable insights. As a tech enthusiast and lifelong learner, I'm always on the hunt for the latest advancements in the ever-evolving landscape of technology. My intrigue with the intricacies of finance has led me to channel my technical prowess into building web app like this website, aiming to educate others in technology and finance."),

            html.P([html.A("LinkedIn", href='https://www.linkedin.com/in/sathya-jayagopi/',
                           target='_blank', className='libutton')], className='home_text'),

            html.P([html.A("GitHub", href='https://github.com/sathjay?tab=repositories',
                           target='_blank', className='libutton')], className='home_text'),

        ], className='description')


    ], className='aboutme_container_first_section'),

    html.Div([
        html.H4('Some great Python Resources from Youtube:'),

        html.Li([html.A("Python for Data Visualization with Dash/Plotly",
                        href='https://www.youtube.com/@CharmingData', target='_blank')], className='anchor_r'),

        html.Li([html.A("Python Programming tutorials",
                        href='https://www.youtube.com/@sentdex', target='_blank')], className='anchor_r'),

        html.Li([html.A("Python for Machine Learning and Data Science",
                        href='https://www.youtube.com/@krishnaik06', target='_blank')], className='anchor_r'),

        html.Li([html.A("Python for Machine Learning with emphasis on Image Processing ",
                        href='https://www.youtube.com/@DigitalSreeni', target='_blank')], className='anchor_r'),

        html.Li([html.A("Python for Quantitative Finance ",
                        href='https://www.youtube.com/@QuantPy', target='_blank')], className='anchor_r'),

        html.Li([html.A("Python for Algo Trading ",
                        href='https://www.youtube.com/@ComputerSciencecompsci112358', target='_blank')], className='anchor_r'),

    ], className='reference')


], className='main_container')
