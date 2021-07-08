# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:21:52 2021

@author: Hamzah
"""
#hist -g   #to recover files
import pandas as pd
#print(os.getcwd())

import os
path = os.path.dirname(os.path.abspath('__file__'))
path = os.path.join(path,"energy.xlsx")
from backend 
energy=pd.read_excel(path,sheet_name='Data')
energy=energy.drop(columns=['Indicator Code','2016'])
#energy.fillna(method='bfill',inplace=True)
#energy.fillna(method='ffill',inplace=True)

#path = path + '\' + "pop.xlsx"
#To get country code
#import geopandas
import geoplot
#world=pd.read_excel('world.xlsx')
import geopandas
#import geoplot
#world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

#world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
list_of_code=world['iso_a3'].tolist()

energy=energy[energy['Country Code'].isin(list_of_code)]
energy.fillna(0,inplace=True)
'''
#
#energy.fillna(method='ffill',inplace=True)

population = pd.read_excel(path,sheet_name='Data',skiprows=[0,1,2])

#dropna(population,# axis=1, how="all", thresh=None, subset=None, inplace=False)
population.dropna(axis=1, how="all", thresh=0.80, inplace=True)
#threshold will keep row with minimum that number of non na 
population.dropna(axis=0, thresh=20, inplace=True)
#population=population.fillna(population.mean())
population=population.fillna(method='bfill',axis=0)
population=population.fillna(method='ffill',axis=0)
'''
col=energy.columns.tolist()
col=col[3:]

last_element_year=int(col[-1])
col_extended= list(range(last_element_year+1,last_element_year+16,1))
temp=col_extended.copy()

temp=col_extended.copy()
col_extended=[]
for i in temp:
    col_extended.append(str(i))
col_extended= col + col_extended
import json
import numpy as np
x=energy.iloc[4]
#x=x[3:]

x=np.asarray(x)
'''
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(x,order=(3,0,1))
model_fit=model.fit()
model_fit.summary()
y=model_fit.predict(start=60,end=70)

subset=energy.iloc[:25]
'''
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource,FactorRange
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.client import push_session, pull_session

from bokeh.models import Slider,Div,Button
from bokeh.layouts import row,column
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)






# Add circle glyphs to the plot

#show(plot)
# Add the plot to the current document and add a title
slider = Slider(start=1990, end=2015, step=1, value=2000, title='Year')
#layout2 = widgetbox(slider)
#curdoc().add_root(layout2)
#################drop down################################
drop=energy['Indicator Name'].unique()
drop=list(drop)
d=[]
for i in drop:
    d.append((i,i))
from bokeh.models import CustomJS, Dropdown,Select
menu=Select(options=drop, value=drop[2],title='Select an Indicator')

#curdoc().add_root(layout)


from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import brewer
from bokeh.plotting import figure



# Input GeoJSON source that contains features for plotting

        
subset_data=energy[energy['Indicator Name']==drop[3]]
subset_data=subset_data[['Country Name','Country Code','Indicator Name','1995']]

col_world=world['iso_a3'].tolist()
col_dataset=subset_data['Country Code'].tolist()
common_col= list(set(col_world).intersection(col_dataset))

world_re= world[world.iso_a3.isin(common_col)]
world_re=world_re.sort_values(by=['iso_a3'])
world_re=world_re.reset_index(drop=True)
world_re=world_re.drop(columns=['pop_est','gdp_md_est'])

re_common=subset_data[subset_data['Country Code'].isin(common_col)]
re_common=re_common.sort_values(by=['Country Code'])


subset= re_common.iloc[:,3]
subset= subset.reset_index()
subset=subset.drop(columns=['index'])

world_re1=pd.concat([world_re,subset],axis=1)
tp=drop[3]
world_re1[tp]=0

world_re1.columns=['Continent','Country_Name','Country Code','geometry','value',tp]
source_map=ColumnDataSource(world_re1)
geosource = GeoJSONDataSource(geojson = world_re1.to_json())


source2=world_re1.sort_values(by=['value'],ascending=False)
source2=source2[0:10]
source2=source2.drop(columns=['geometry'])
source2=source2.reset_index()
source_2=ColumnDataSource(source2)
#rangex=source_2.data['Country_Name']
#rangex=list(rangex)
f=list(source_2.data['Country_Name'])
#bar.x_range=(f)
bar = figure(x_range=f,plot_height=300, title="Top 10 countries",
           toolbar_location=None, tools="")
#f=FactorRange(factors=list(source_2.data['Country_Name']))

bar.vbar(x='Country_Name', top='value', width=0.3,source=source_2)
#show(bar)

bar.xgrid.grid_line_color = None
bar.y_range.start = 0
bar.xaxis.axis_label = "Countries"
bar.xaxis.major_label_orientation = 1.2
bar.outline_line_color = None

#show(bar)


# Define color palettes
palette = brewer['BuGn'][8]
palette = palette[::-1] # reverse order of colors so higher values have darker colors
# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
high_value=source_map.data['value'].max()
name_ind=source_map.data

#high_value=world_re1['value'].max()
color_mapper = LinearColorMapper(palette = palette)

# Define custom tick labels for color bar.

# Create color bar.
color_bar = ColorBar(color_mapper = color_mapper, 
                     label_standoff = 8,
                     width = 500, height = 20,
                     border_line_color = None,
                     location = (0,0), 
                     orientation = 'horizontal')
# Create figure object.
p = figure(title = 'Global Energy Indiator', 
           plot_height = 600, plot_width = 950, 
           toolbar_location = 'below',
           tools = "pan, wheel_zoom, box_zoom, reset")
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
# Add patch renderer to figure.
country = p.patches('xs','ys', source = geosource,
                   fill_color = {'field' :'value',
                                 'transform' : color_mapper},
                   line_color = 'gray', 
                   line_width = 0.25, 
                   fill_alpha = 1)

# Create hover tool
p.add_tools(HoverTool(renderers = [country],
                      tooltips = [('Country','@Country_Name'),
                                ('value','@value')]))

# Specify layout
p.add_layout(color_bar, 'below')
#show(p)


####line plot
unique_country=list(world_re1['Country_Name'].unique())
subset_line_data=energy.copy()
subset_line_data=subset_line_data[subset_line_data['Country Name']==unique_country[5]]
subset_line_data=subset_line_data[subset_line_data['Indicator Name']==drop[5]]

x_subset_line=subset_line_data.columns.tolist()
x_subset_line=x_subset_line[3:]
temp_list=[]
for i in x_subset_line:
    temp_list.append(int(i))
x_subset_line=temp_list.copy()  
y_subset_line=list(subset_line_data.reset_index().iloc[0])
y_subset_line=y_subset_line[4:]

#source3=ColumnDataSource(data={'x_values':x_subset_line,'y_values':y_subset_line})

select1=Select(options=unique_country, value=unique_country[5],title='Select Country')
select3=Select(options=drop, value=drop[5],title='Select an Indicator')


line = figure(plot_width=400, plot_height=400,x_axis_label='years')

line.line(y='y_values', x='x_values', source=source3)

#show(line)


def update_lineplot():
    country_name=select2.value
    year=select3.value
    subset_line=energy.copy()
    subset_line=subset_line[subset_line['Country Name']==country_name]
    subset_line=subset_line[subset_line['Indicator Name']==year]
    
    x_subset=subset_line.columns.tolist()
    x_subset=x_subset[3:]
    temp_list=[]
    for i in x_subset:
        temp_list.append(int(i))
    x_subset=temp_list.copy()   
    y_subset=list(subset_line.reset_index().iloc[0])
    y_subset=y_subset[4:]
    data={'x_values':x_subset,'y_values':y_subset}
    source3.data=data

button = Button(label="See graph", button_type="success")
button.on_click(update_lineplot)
layout_line=column(select2,select3,button,line)

#show(dropdown)
###############################################################
#curdoc().add_root(p)
#curdoc().title = 'Gapminder'

########################callback
def callback(attr,old,new):
    N=slider.value
    year=str(N)
    indicator=menu.value
    subset_data=energy[energy['Indicator Name']==indicator]
    subset_data=subset_data=subset_data[['Country Name','Country Code','Indicator Name',year]]
    
    col_world=world['iso_a3'].tolist()
    col_dataset=subset_data['Country Code'].tolist()
    common_col= list(set(col_world).intersection(col_dataset))
    
    world_re= world[world.iso_a3.isin(common_col)]
    world_re=world_re.sort_values(by=['iso_a3'])
    world_re=world_re.reset_index()
    world_re=world_re.drop(columns=['index','pop_est','gdp_md_est'])
    
    re_common=subset_data[subset_data['Country Code'].isin(common_col)]
    re_common=re_common.sort_values(by=['Country Code'])
    
    
    subset= re_common.iloc[:,3]
    subset= subset.reset_index()
    subset=subset.drop(columns=['index'])
    
    world_re1=pd.concat([world_re,subset],axis=1)
    world_re1[indicator]=0
    world_re1.columns=['Continent','Country_Name','Country Code','geometry','value',indicator]
    source_map.data=world_re1
    source_inside2=world_re1.sort_values(by=['value'],ascending=False)
    source_inside2=source_inside2[0:10]
    source_inside2=source_inside2.drop(columns=['geometry'])
    source_2.data=source_inside2
    bar.x_range.factors=list((source_2.data['Country_Name']))
    #bar.x_range= list(source_2.data['Country_Name'])
    geosource.geojson = world_re1.to_json() 
    #GeoJSONDataSource(geojson = world_re2.to_json())

#show(menu)
#output_file("slider.html")
title = Div(text='<h1 style="width: 2000px;background-color:LightGray;font-size:40px;font-family:garamond"> Prognosis and visualization of Global Energy Indicator database</h1>')
title2 = Div(text='<p style="width: 2000px;text-align:left;font-size:10px"> Developer: Hamzah Shabbir </p>')
title3 = Div(text='<h3 style="text-align:center;background-color:LightGray;font-size:10px"> Using Arima model for Forecasting of Global Energy statistics and Visualisations</h1>')
#show(title)
#show(title)
tit_lay=column(title,title2)
layout_x=column(slider,menu)
layout_y=row(layout_x,bar)
p_layout=row(p,layout_x)

layout_all=column(children=[tit_lay,p_layout,layout_y,layout_line])
#show(title)
slider.on_change('value',callback)
menu.on_change('value',callback)
layout1=row(slider, p)
layout1=row(children=[p,bar])
layout2=row(slider,menu)
layout3=column(children=[layout1,layout2])
#layout4=row(widgetbox(slider, menu),bar,p_layout,layout_line)
curdoc().add_root(layout_all)
