# Import required libraries
import pathlib
import math
from urllib.request import urlopen
import json
import re
import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import plotly_express as px
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
import datetime as dt
from datetime import timedelta
from datetime import date
import numpy as np
import dash_daq as daq
from dateutil import relativedelta


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server  # the Flask app


########################################
########## Data Preprocessing ##########
# Read data
retailer_df = pd.read_csv(DATA_PATH.joinpath("retailer.csv"))
# retailer_df = retailer_df.sort_values("Siparis_Tarihi").set_index("Siparis_Tarihi")

tr_sehirler = json.load(open(DATA_PATH.joinpath(
    "tr-cities-utf8_plaka_kodu.json"), 'r'))  # harita verisi

lookup_df = pd.read_csv(DATA_PATH.joinpath(
    "tr_iller_plaka_kodu.csv"), dtype={"number": str})
#print(lookup_df.head(10))

retailer_df['Plaka_kodu'] = retailer_df['Sehir'].map(
    lookup_df.set_index('Il')['Plaka_kodu'])
#print(retailer_df.head(10))

# Satis Miktarı Sütunu
retailer_df["Siparis_Miktari"] = 1

# Date
# Format siparis zamanı
retailer_df["Siparis_Tarihi"] = retailer_df["Siparis_Tarihi"].apply(
    lambda x: datetime.strptime(x, "%d/%m/%Y")
)  # String -> Datetime


# Date
# Format siparis zamanı
retailer_df["Sevk_Tarihi"] = retailer_df["Sevk_Tarihi"].apply(
    lambda x: datetime.strptime(x, "%d/%m/%Y")
)

# Insert weekday, month, and year of siparis tarihi
retailer_df["Days_of_Wk"] = retailer_df["Siparis_Ayi"] = retailer_df["Siparis_Yili"] = retailer_df["Siparis_Tarihi"]

retailer_df["Days_of_Wk"] = retailer_df["Days_of_Wk"].apply(
    lambda x: datetime.strftime(x, "%A")
)  # Datetime -> weekday string

retailer_df["Siparis_Ayi"] = retailer_df["Siparis_Ayi"].apply(
    lambda x: datetime.strftime(x, "%m")
)  # Datetime -> month

retailer_df["Siparis_Yili"] = retailer_df["Siparis_Yili"].apply(
    lambda x: datetime.strftime(x, "%Y")
)  # Datetime -> month

day_list = [
    "Sunday",
    "Saturday",
    "Friday",
    "Thursday",
    "Wednesday",
    "Tuesday",
    "Monday",
]

kategori_list = retailer_df["Kategori"].unique().tolist()
segment_list = retailer_df["Segment"].unique().tolist()
########## Data Preprocessing ##########
########################################

###############################
########## layout #############

app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("datajarlabs-logo.png"),
                            id="logo-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Perakende Satış Analizi",
                                    style={"margin-bottom": "0px",
                                           "color": "blue"},
                                ),
                                html.H5(
                                    "Dashboard", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Detaylı Bilgi",
                                        id="learn-more-button"),
                            href="https://datajarlabs.com/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(children="Tarih aralığı seçiniz :",
                               style={"font-weight": "bold", }),
                        dcc.RangeSlider(id="time-window-slider"),
                        html.Br(),
                        html.P("Gösterim türünü seçiniz:", className="control_label", style={
                               "font-weight": "bold"}),
                        dcc.RadioItems(
                            id="gosterim_status_selector",
                            options=[
                                {"label": "Sipariş", "value": "siparis"},
                                {"label": "Satış", "value": "satis"},
                                {"label": "Kâr", "value": "kar"},
                            ],
                            value="satis",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        html.Br(),
                        html.P(children="Segment seçiniz :",
                               style={"font-weight": "bold"}),
                        dcc.Dropdown(
                            id="segment-select",
                            options=[{"label": i, "value": i}
                                     for i in segment_list],
                            value=segment_list,
                            multi=True,
                        ),
                        html.Br(),
                        html.P(children="Kategori seçiniz :",
                               style={"font-weight": "bold"}),
                        dcc.Dropdown(
                            id="kategori-select",
                            options=[{"label": i, "value": i}
                                     for i in kategori_list],
                            value=kategori_list,
                            multi=True,
                        ),
                        html.Br(),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="siparisText"),
                                     html.P("Sipariş")],
                                    id="siparis",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="satisText"), html.P("Satış")],
                                    id="satis",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="karText"), html.P("Kâr")],
                                    id="kar",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [dcc.Graph(id="tr_choropleth")],
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="bolge_histogram")],
                    #style={'height':'200'},
                    id="countGraphContainer",
                    className="pretty_container seven columns",
                ),

                html.Div(
                    [dcc.Graph(id="individual_graph")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="bolge-pie")],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="kategori-pie")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    id="heatmap_card",
                    children=[
                        dcc.Graph(id="heatmap"),  # patient_volume_hm
                    ],
                    className="pretty_container twelwe columns",
                ),
            ],
            className="row flex-display",
        )
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

########## layout #############
###############################


###############################
######### functions ###########
def comma_me(amount):
    orig = amount
    new = re.sub("^(-?\d+)(\d{3})", '\g<1>,\g<2>', amount)
    if orig == new:
        return new
    else:
        return comma_me(new)


def human_format(num):
    if num == 0:
        return "0"

    magnitude = int(math.log(num, 1000))
    mantissa = str(int(num / (1000 ** magnitude)))
    return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]


def generate_pie_graph(start_date, end_date, column, selector, segment, kategori):

    filtered_df = retailer_df.sort_values("Siparis_Tarihi").set_index("Siparis_Tarihi")[
        start_date:end_date
    ]

    filtered_df = filtered_df[filtered_df["Segment"].isin(
        segment) & filtered_df["Kategori"].isin(kategori)]

    nb_cases = len(filtered_df.index)
    types = []
    values = []
    title1 = ""
    title2 = ""

    if column == "Bolge":
        title1 = "Bölgelere Göre "
    else:
        title1 = "Kategorilere Göre "

    types = filtered_df[column].unique().tolist()

    # if no results were found
    if types == []:
        layout = dict(
            autosize=True,
            annotations=[dict(text="No results found", showarrow=False)]
        )
        return {"data": [], "layout": layout}

    if selector == "siparis":
        title2 = "Sipariş"
        for type in types:
            nb_type = filtered_df.loc[filtered_df[column] == type].shape[0]
            values.append(nb_type / nb_cases * 100)
    elif selector == "satis":
        title2 = "Satış"
        total_satis = filtered_df["Satis"].sum()
        for type in types:
            type_satis = filtered_df.loc[filtered_df[column]
                                         == type]["Satis"].sum()
            values.append(type_satis/total_satis*100)
    else:
        title2 = "Kâr"
        total_kar = filtered_df["Kar"].sum()
        for type in types:
            type_kar = filtered_df.loc[filtered_df[column]
                                       == type]["Kar"].sum()
            values.append(type_kar/total_kar*100)

    crit = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    color_dict = {'1': '#e6f2ff',
                  '2': '#99ccff',
                  '3': '#ccccff',
                  '4': '#cc99ff',
                  '5': '#ff99ff',
                  '6': '#ff6699',
                  '7': '#ff9966',
                  '8': '#ff6600',
                  '9': '#ff5050',
                  '10': '#ff0000'}

    colors = np.array([''] * len(crit), dtype=object)

    for i in np.unique(crit):
        colors[np.where(crit == i)] = color_dict[str(i)]

    layout = go.Layout(
        title=title1 + title2 + " Oranları",
        autosize=True,
        margin=dict(l=10, r=10, b=40, t=40, pad=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=400,
    )

    trace = go.Pie(
        labels=types,
        values=values,
        marker={"colors": colors},
        hoverinfo="value",  # text+value+percent
        textinfo="label+percent",  # label+percent+name
        hole=0.1,

    )

    return {"data": [trace], "layout": layout}


def generate_bolge_histogram(start_date, end_date, selector, segment, kategori):

    filtered_df = retailer_df.sort_values("Siparis_Tarihi").set_index(
        "Sevk_Tarihi")[start_date:end_date]

    filtered_df = filtered_df[filtered_df["Segment"].isin(
        segment) & filtered_df["Kategori"].isin(kategori)]

    if selector == "siparis":
        y_val = "Siparis_Miktari"
        yaxis_title = "Sipariş Miktarı"
        hist_func = 'count'
    elif selector == "satis":
        y_val = "Satis"
        yaxis_title = "Satış"
        hist_func = 'sum'
    else:
        y_val = "Kar"
        yaxis_title = "Kâr"
        hist_func = 'sum'

    figure = px.histogram(filtered_df,
                          x="Siparis_Tarihi",
                          y=y_val,
                          color='Bolge',
                          color_discrete_sequence=[
                              '#c6dbef',
                              '#9ecae1',
                            '#6baed6',
                            '#4292c6',
                            '#2171b5',
                            '#08519c',
                            '#08306b',
                          ],
                          height=400,
                          histfunc=hist_func,
                          #marginal='box',
                          )

    figure.update_layout(
        autosize=True,
        margin=dict(l=40, r=10, b=10, t=20),
        hovermode="closest",
        xaxis_title_text='Sipariş Dönemi',  # xaxis label
        yaxis_title_text=yaxis_title,  # yaxis label
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )

    return figure


def make_marks_time_slider(mini, maxi):
    """
    A helper function to generate a dictionary that should look something like:
    {1420066800: '2015', 1427839200: 'Q2', 1435701600: 'Q3', 1443650400: 'Q4',
    1451602800: '2016', 1459461600: 'Q2', 1467324000: 'Q3', 1475272800: 'Q4',
     1483225200: '2017', 1490997600: 'Q2', 1498860000: 'Q3', 1506808800: 'Q4'}
    """
    step = relativedelta.relativedelta(months=+1)
    start = datetime(year=mini.year, month=1, day=1)
    #print("\tmini_start : ", start)
    end = datetime(year=maxi.year, month=maxi.month, day=30)
    #print("\tmaxi_end : ", end)
    ret = {}

    current = start
    while current <= end:
        current_str = int(current.timestamp())
        if current.month == 1:
            ret[current_str] = {
                "label": str(current.year),
                "style": {"font-weight": "bold"},
            }
        elif current.month == 4:
            ret[current_str] = {
                "label": "Q1",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        elif current.month == 7:
            ret[current_str] = {
                "label": "Q2",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        elif current.month == 10:
            ret[current_str] = {
                "label": "Q3",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        else:
            pass
        current += step
    # print(ret)
    return ret


def generate_heatmap(start_date, end_date, selector, segment, kategori):
    """
    :param: start: start date from selection.
    :param: end: end date from selection.
    """

    filtered_df = retailer_df.sort_values("Siparis_Tarihi").set_index("Siparis_Tarihi")[
        start_date:end_date
    ]

    filtered_df = filtered_df[filtered_df["Segment"].isin(
        segment) & filtered_df["Kategori"].isin(kategori)]

    x_axis = []
    for i in range(1, 13):
        x_axis.append((dt.date(2018, i, 1).strftime('%m')))

    y_axis = day_list

    month = ""
    weekday = ""

    if selector == "siparis":
        hovertemplate = "<b> %{y}  %{x} <br><br> %{z} Adet Sipariş"
        record = "Siparis_Miktari"
        title = "Sipariş Miktarları"
    elif selector == "satis":
        hovertemplate = "<b> %{y}  %{x} <br><br> %{z} TL Satış"
        record = "Satis"
        title = "Satış Tutarları"
    else:
        hovertemplate = "<b> %{y}  %{x} <br><br> %{z} TL Kâr"
        record = "Kar"
        title = "Kâr Tutarları"

    # Get z value : sum(number of records) based on x, y,

    z = np.zeros((7, 12))
    annotations = []

    for ind_y, day in enumerate(y_axis):
        filtered_day = filtered_df[filtered_df["Days_of_Wk"] == day]

        for ind_x, x_val in enumerate(x_axis):
            sum_of_record = filtered_day[filtered_day["Siparis_Ayi"] == x_val][record].sum(
            )
            sum_of_record = f'{sum_of_record:.0f}'
            z[ind_y][ind_x] = sum_of_record

            annotation_dict = dict(
                showarrow=False,
                text="<b>" + str(sum_of_record) + "<b>",
                xref="x",
                yref="y",
                x=x_val,
                y=day,
                font=dict(family="sans-serif"),
            )

            annotations.append(annotation_dict)

    data = [
        dict(
            x=x_axis,
            y=y_axis,
            z=z,
            type="heatmap",
            name="",
            hovertemplate=hovertemplate,
            showscale=True,
            colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
            #colorscale=[[0, "#fdffca"], [1, "#02f502"]],
            #colorscale='Jet',
        )
    ]

    layout = dict(
        title=f"Gün-Ay Bazında {title}",
        margin=dict(l=70, b=20, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="sans-serif"),
        annotations=annotations,
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left",
            ticks="",
            tickfont=dict(family="sans-serif"),
            ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
    )
    return {"data": data, "layout": layout}


def time_slider_to_date(time_values):
    """ TODO """
    min_date = datetime.fromtimestamp(time_values[0]).strftime("%c")
    max_date = datetime.fromtimestamp(time_values[1]).strftime("%c")
    return [min_date, max_date]


######### functions ###########
###############################

# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("bolge_histogram", "figure")],
)


######################################
############  callbacks  #############


@app.callback(
    Output("bolge-pie", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("gosterim_status_selector", "value"),
        Input("segment-select", "value"),
        Input("kategori-select", "value"),
    ],
)
def update_bolge_pie(time_values, selector, segment, kategori):

    graph_type = "Bolge"

    if time_values is not None:
        min_date, max_date = time_slider_to_date(time_values)
        return generate_pie_graph(min_date, max_date, graph_type, selector, segment, kategori)

    return {"data": []}


@app.callback(
    Output("kategori-pie", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("gosterim_status_selector", "value"),
        Input("segment-select", "value"),
        Input("kategori-select", "value"),
    ],
)
def update_kategori_pie(time_values, selector, segment, kategori):

    graph_type = "Kategori"

    if time_values is not None:
        min_date, max_date = time_slider_to_date(time_values)
        return generate_pie_graph(min_date, max_date, graph_type, selector, segment, kategori)

    return {"data": []}


@app.callback(
    Output("bolge_histogram", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("gosterim_status_selector", "value"),
        Input("segment-select", "value"),
        Input("kategori-select", "value"),
    ],
)
def update_bolge_histogram(time_values, selector, segment, kategori):
    """
    Depending on our dataset, we need to draw the initial histogram.
    """

    if time_values is not None:
        min_date, max_date = time_slider_to_date(time_values)
        return generate_bolge_histogram(min_date, max_date, selector, segment, kategori)

    return {"data": []}


@app.callback(
    [
        Output("time-window-slider", "marks"),
        Output("time-window-slider", "min"),
        Output("time-window-slider", "max"),
        Output("time-window-slider", "step"),
        Output("time-window-slider", "value"),
    ],
    [Input("logo-image", "src")],
)
def populate_time_slider(src):
    """
    Depending on our dataset, we need to populate the time-slider
    with different ranges. This function does that and returns the
    needed data to the time-window-slider.
    """

    min_date = retailer_df["Siparis_Tarihi"].min()
    max_date = retailer_df["Siparis_Tarihi"].max()

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )


@app.callback(
    Output("tr_choropleth", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("gosterim_status_selector", "value"),
        Input("segment-select", "value"),
        Input("kategori-select", "value"),
    ],
)
def generate_choropleth(time_values, selector, segment, kategori):

    if time_values is not None:
        min_date, max_date = time_slider_to_date(time_values)

        filtered_df = retailer_df.sort_values("Siparis_Tarihi").set_index("Siparis_Tarihi")[
            min_date:max_date]

        filtered_df = filtered_df[filtered_df["Segment"].isin(
            segment) & filtered_df["Kategori"].isin(kategori)]

        df = pd.DataFrame(filtered_df.groupby(
            ["Plaka_kodu"]).sum().reset_index())
        # print(df.head(10))
        # hovertemplate = "<br> %{z} TL"

        if selector == "siparis":
            z_val = "Siparis_Miktari"
        elif selector == "satis":
            z_val = "Satis"
        else:
            z_val = "Kar"

        fig = go.Figure(go.Choroplethmapbox(geojson=tr_sehirler, locations=df.Plaka_kodu, z=df[z_val],
                                            colorscale=[
                                                [0, "#65aaf7"], [1, "#012d6b"]],
                                            colorbar=dict(
                                                thickness=20, ticklen=3),
                                            #hovertemplate = hovertemplate,
                                            #hoverinfo=z,
                                            zmin=df[z_val].min(), zmax=df[z_val].max(),
                                            below=True,
                                            marker_opacity=0.6, marker_line_width=0.2))

        fig.update_layout(mapbox_style="carto-positron",
                          mapbox_zoom=4.5, mapbox_center={"lat": 39.822060, "lon": 34.808132})

        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        return fig

    return {"data": []}


@app.callback(
    [
        Output("siparisText", "children"),
        Output("satisText", "children"),
        Output("karText", "children"),
    ],
    [
        Input("time-window-slider", "value"),
        Input("segment-select", "value"),
        Input("kategori-select", "value"),
    ],
)
def update_siparis_text(time_values, segment, kategori):
    if time_values is not None:
        min_date, max_date = time_slider_to_date(time_values)

        filtered_df = retailer_df.sort_values("Siparis_Tarihi").set_index("Siparis_Tarihi")[
            min_date:max_date]

        filtered_df = filtered_df[filtered_df["Segment"].isin(
            segment) & filtered_df["Kategori"].isin(kategori)]

        siparis = f'{filtered_df.Siparis_Miktari.sum():.0f}'
        siparis = comma_me(siparis)

        satis = f'{filtered_df.Satis.sum():.0f}'
        satis = comma_me(satis)

        kar = f'{filtered_df.Kar.sum():.0f}'
        kar = comma_me(kar)

        result = siparis + " adet", satis + " TL", kar + " TL"
    else:
        result = "no result", "no result", "no result"

    return result


@app.callback(
    Output("individual_graph", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("segment-select", "value"),
        Input("kategori-select", "value"),
    ],
)
def make_individual_figure(time_values, segment, kategori):

    if time_values is not None:
        min_date, max_date = time_slider_to_date(time_values)

        filtered_df = retailer_df.sort_values("Siparis_Tarihi").set_index("Siparis_Tarihi")[
            min_date:max_date]

        filtered_df = filtered_df[filtered_df["Segment"].isin(
            segment) & filtered_df["Kategori"].isin(kategori)]

        resamp = filtered_df.resample("M").sum().reset_index()

        data = [
            dict(
                type="scatter",
                mode="lines+markers",
                name="Satış (TL)",
                x=resamp["Siparis_Tarihi"],
                y=resamp["Satis"],
                line=dict(shape="spline", smoothing=2,
                          width=1, color="#08306b"),
                marker=dict(symbol="diamond-open"),
            ),
            dict(
                type="scatter",
                mode="lines+markers",
                name="Kâr (TL)",
                x=resamp["Siparis_Tarihi"],
                y=resamp["Kar"],
                line=dict(shape="spline", smoothing=2,
                          width=1, color="#4292c6"),
                marker=dict(symbol="diamond-open"),
            ),
        ]

        layout = dict(
            autosize=True,
            automargin=True,
            margin=dict(l=40, r=10, b=10, t=50),
            hovermode="closest",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(font=dict(size=10), orientation="h"),
            height=350,
        )

        figure = dict(data=data, layout=layout)
        return figure

    return {"data": []}


@app.callback(
    Output("heatmap", "figure"),
    [
        Input("time-window-slider", "value"),
        Input("gosterim_status_selector", "value"),
        Input("segment-select", "value"),
        Input("kategori-select", "value"),
    ],
)
def update_heatmap(time_values, selector, segment, kategori):
    #print("\t time values first heatmap : type ", type(time_values))
    if time_values is not None:
        # Return to original hm(no colored annotation) by resetting
        min_date, max_date = time_slider_to_date(time_values)
        return generate_heatmap(min_date, max_date, selector, segment, kategori)

    return {"data": []}

######## callbacks #########
############################


# Main
if __name__ == "__main__":
    app.run_server(debug=True)
