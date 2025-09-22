import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import zscore
import streamlit as st

#import plotly.express as px

from utils.sentences import format_metric
from classes.data_point import Player, Country, Person
from classes.data_source import PlayerStats, CountryStats, PersonStat
from typing import Union


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"

def wrap_text(text, max_len=15):
    words = text.split()
    wrapped_text = ""
    current_len = 0
    for word in words:
        if current_len + len(word) > max_len:
            wrapped_text += "<br>"
            current_len = 0
        wrapped_text += word + " "
        current_len += len(word) + 1
    return wrapped_text.strip()

class DistributionPlot:
    def __init__(self, dataframe, entity, metrics, *args, **kwargs):
        self.cols = metrics
        self.dataframe = dataframe.df
        self.entity = entity.ser_metrics
        self.background = hex_to_rgb("#faf9ed")
        self.color = hex_to_rgb("#eddefa")
        self.fig = go.Figure()
        self.set_visualization()
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            use_container_width=True,
        )

    def _setup_axes(self):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            showgrid=False,
            gridcolor=rgb_to_color(hex_to_rgb("#6a5acd"), 0.7),
        )
        self.fig.update_yaxes(
            showticklabels=True,
            fixedrange=True,
            showgrid=False,
            gridcolor=rgb_to_color(self.background),
            zerolinecolor=rgb_to_color(hex_to_rgb("#ffffff")),
        )



    def set_visualization(self):
        dataframe = self.dataframe.iloc[:, -2*len(self.cols):-len(self.cols)]
        df_entity = self.entity.iloc[-2*len(self.cols):-len(self.cols)]
        df_entity_rank = self.entity.iloc[-len(self.cols):]

        # Color palette
        colors = px.colors.qualitative.Set2

        # Create subplots
        self.fig = make_subplots(
            rows=len(self.cols),
            cols=1,
            shared_xaxes=True,  # Keep the same scale for all
            vertical_spacing=0.0
        )

        for i, col in enumerate(dataframe.columns):
            self.fig.add_trace(
                go.Violin(
                    x=dataframe[col].tolist(),
                    name = self.cols[i],
                    #name=wrap_text(self.cols[i]),
                    marker=dict(color=colors[i % len(colors)]),
                    opacity=0.65,
                    side='positive',
                    showlegend = False,
                    hovertemplate=f"<b>{self.cols[i]}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ),
                row=i+1,
                col=1
            )

            # Entity marker
            entity_value = float(df_entity.iloc[i])
            self.fig.add_trace(
                go.Scatter(
                    x=[entity_value],
                    y=[self.cols[i]],
                    mode="markers", # if we want marker and text do "markers+text"
                    marker=dict(symbol="diamond", size=6, color="#9340ff"),
                    name="Selected entity",  # this will appear in the legend
                    showlegend=(i == 0),  # ensures legend is shown
                    hovertemplate=f"<b>{self.cols[i]}</b><br>Value: %{{x}}<br>Rank: %{{customdata}}<extra></extra>",
                    customdata=[round(float(df_entity_rank.iloc[i]))]
                
                    ),
                    row=i+1,
                    col=1
                    )


        # Update layout
        self.fig.update_layout(
            template="plotly_white",
            title=dict(text="<b>Distribution of Metrics</b>",x=0.55, font=dict(size=14)),
            showlegend=True,
            margin=dict(t=50, b=50, l=45, r=25),
            font = dict(size=14),
            autosize=True,
            legend=dict(
                yanchor="bottom",
                y=-0.2,
                xanchor="right",
                x=1,
                font=dict(size=10)
            )
        )

    
        # Add grid & font styling
        self.fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        self.fig.update_yaxes(showgrid=False)




class RadarPlot:
    def __init__(self, entity, metrics, *args, **kwargs):

        self.cols = metrics
        self.entity = entity.ser_metrics
        self.color = hex_to_rgb("#faf9ed")
        self.fig = go.Figure()
        self.set_visualization()

        super().__init__(*args, **kwargs)
     

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            use_container_width=True,
        )


    def set_visualization(self):
        # Streamlit primary color
        color = st.get_option("theme.primaryColor")
        if color is None:
            color = "#FF4B4B"

    
        df_entity = self.entity.iloc[-2*len(self.cols):-len(self.cols)]
        r_values = df_entity.values.tolist()
        #theta_values=self.cols
        theta_values=[wrap_text(c) for c in self.cols]

        # Repeat the first element at the end to close the polygon
        r_values.append(r_values[0])
        theta_values = theta_values + [theta_values[0]]

        # Add the entity as a highlighted polygon
        self.fig.add_trace(
            go.Scatterpolar(
                r = r_values,
                theta = theta_values,
                mode="lines+markers",
                line=dict(color=rgb_to_color(hex_to_rgb("#9340ff")), width=3),
                marker=dict(size=8, color=rgb_to_color(hex_to_rgb("#9340ff"))),
                fill="toself",
                hovertemplate="<b>%{theta}</b>: %{r}<extra></extra>", 
                showlegend=False,
            )
        )


        self.fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-4, 4],
                    gridcolor=rgb_to_color(hex_to_rgb("#E0E0E0")),
                    linecolor=rgb_to_color(hex_to_rgb("#CCCCCC")),
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, family="Gilroy-Light", color="#333")
                )
            ),
            margin=dict(l=75, r=85, t=55, b=55),
            plot_bgcolor="white",
            showlegend=False,
            autosize=True,
        )




