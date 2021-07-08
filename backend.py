# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:24:24 2021

@author: Hamzah
"""
#hist -g   #to recover files
import pandas as pd
#print(os.getcwd())

import os
path= os.path.dirname(os.path.abspath('__file__'))
path = os.path.join(path,"energy.xlsx")

energy=pd.read_excel(path,sheet_name='Data')
energy=energy.drop(columns=['Indicator Code','2016'])
#energy.fillna(method='bfill',inplace=True)
#energy.fillna(method='ffill',inplace=True)

#path = path + '\' + "pop.xlsx"
#To get country code
import geopandas
#import geoplot
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
list_of_code=world['iso_a3'].tolist()

energy=energy[energy['Country Code'].isin(list_of_code)]
energy.fillna(0,inplace=True)

col=energy.columns.tolist()
col=col[3:]

last_element_year=int(col[-1])
col_extended= list(range(last_element_year+1,last_element_year+16,1))
temp1=col_extended.copy()

#temp=col_extended.copy()
col_extended=[]
for i in temp1:
    col_extended.append(str(i))
col_extended= col + col_extended

from scipy.optimize import curve_fit
def optm(x, a, b):
    return a*x+b
    



import numpy as np
x=energy.iloc[5]
x=x[3:]
x=np.array(x)
y=[]
for i in col:
    y.append(int(i))
    
y=np.array(y)
x_axis=y.copy()
y_axis=x.copy()
param, pcov = curve_fit(optm, x_axis, y_axis)
a,b=param
x_next=[]
y_next=[]
for i in temp1:
    #x_next.append(int(i))
    temp=optm(int(i),a,b)
    y_next.append(temp)
    
y_total=list(y_axis) + y_next

energy.to_excel('b.xlsx')
df=energy[0:15]
y=[]
for i in col:
    y.append(int(i))
y=np.array(y)
x_axis=y.copy()

df_final=[]
for i in range(len(df)):
    x=df.iloc[i]
    x=x[3:]
    x=np.array(x)
    
    y_axis=x.copy()
    param, pcov = curve_fit(optm, x_axis, y_axis)
    a,b=param
    x_next=[]
    y_next=[]
    for i in temp1:
        #x_next.append(int(i))
        temp=optm(int(i),a,b)
        y_next.append(temp)
        
    y_total=list(y_axis) + y_next
    df.append(y_total)
    

    
    



import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float64')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    #print("Evaluating the settings: ", p, d, q)
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #print('ARIMA%s MSE=%.3f' % (order,mse))
                except Exception as exception:
                    pass
                   
    return best_cfg

df_final=[]
for i in range(len(df)):
    x=df.iloc[i]
    x=x[18:]
    x=np.asarray(x)
    p_values=range(1,3)
    d_values=range(0,1)
    q_values=range(0,3)
    order=evaluate_models(x,p_values,d_values,q_values)
    if order == None:
        df_final.append(0)
    else:
        model_fit=ARIMA(x,order=order).fit()
        model_fit=ARIMA(x,order=order).fit()
        y_predict=model_fit.predict(start=5,end=41)
        #y_predict=list(x)+list(y_predict)
        df_final.append(y_predict)

        
    model_fit=ARIMA(x,order=order).fit()
    y_predict=model_fit.predict(start=26,end=41)
    y_predict=list(x)+list(y_predict)
    df_final.append(y_predict)
    
    for p in range(1,3):
        for d in range(0,1):
            for q in range(0,3):
                try:
                    model=ARIMA(x,order=(p,d,q))

                    model_fit=model.fit()
                    break
                except ValueError:
                    continue
                except LinAlgError:
                    continue
    
    y_predict=model_fit.predict(start=26,end=41)
    y_predict=list(x)+list(y_predict)
    df_final.append(y_predict)
    k=pd.DataFrame(df.final)
    
    try:
        model_fit=model.fit()
    except ValueError:
        p=p+1
        q=q+1
        d=d+1
        try:
            model_fit=model.fit()
        except ValueError:
            p=p+1
            model_fit=model.fit()
            
            
            

        
    except AttributeError:
        try:
            return self.dict[item]
        except KeyError:
            print "The object doesn't have such attribute"
    model_fit=model.fit()
    #model_fit.summary()
    
    
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
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import Slider
from bokeh.layouts import row,column


source = ColumnDataSource(data={
    'y'       : energy['1991'],
    'x'       : energy['1992']
    
})
ymin, ymax = min(energy['1991']), max(energy['1991'])
xmin, xmax = min(energy['1992']), max(energy['1992'])

plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'
#show(plot)
# Add the plot to the current document and add a title
slider = Slider(start=1990, end=2015, step=1, value=2000, title='Year')
layout = widgetbox(slider)
curdoc().add_root(layout)
#################drop down################################
drop=energy['Indicator Name'].unique()
drop=list(drop)
d=[]
for i in drop:
    d.append((i,i))
from bokeh.models import CustomJS, Dropdown,Select
menu=Select(options=drop, value=drop[2],title='Select an Indicator')

layout=column(menu, plot)
#curdoc().add_root(layout)
'''dropdown = Dropdown(label="Select an Indicator", button_type="warning", menu=d)
dropdown.js_on_event("menu_item_click", CustomJS(code="console.log('dropdown: ' + this.item, this.toString())"))
layout=column(dropdown, plot)
#curdoc().add_root(layout)'''
############################################################################
######################################################################
############# #map#############################



import json
from bokeh.io import show
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import brewer
from bokeh.plotting import figure
# Input GeoJSON source that contains features for plotting
'''def map_plotting(data):
    geosource = GeoJSONDataSource(geojson = data.to_json())
    palette = brewer['BuGn'][8]
    palette = palette[::-1] # reverse order of colors so higher values have darker colors
    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    
    high_value=data['value'].max()
    color_mapper = LinearColorMapper(palette = palette, low = 0, high = high_value)
    

    color_bar = ColorBar(color_mapper = color_mapper, 
                     label_standoff = 8,
                     width = 500, height = 20,
                     border_line_color = None,
                     location = (0,0), 
                     orientation = 'horizontal')
    # Create figure object.
    p = figure(title = 'Lead Levels in Water Samples, 2018', 
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
                                    ('Population','@value')]))
    
    # Specify layout
    p.add_layout(color_bar, 'below')'''
        
subset_data=energy[energy['Indicator Name']==drop[3]]
subset_data=subset_data=subset_data[['Country Name','Country Code','Indicator Name','1995']]

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
world_re1[drop[3]]=0
world_re1.columns=['Continent','Country_Name','Country Code','geometry','value',drop[3]]

##################################################
###################################################
#####################################################
#hist -g   #to recover files
import pandas as pd
#print(os.getcwd())

import os
path= os.path.dirname(os.path.abspath('__file__'))
path = os.path.join(path,"energy.xlsx")

energy=pd.read_excel(path,sheet_name='Data')
energy=energy.drop(columns=['Indicator Code','2016'])
#energy.fillna(method='bfill',inplace=True)
#energy.fillna(method='ffill',inplace=True)

#path = path + '\' + "pop.xlsx"
#To get country code
import geopandas
#import geoplot
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
list_of_code=world['iso_a3'].tolist()

energy=energy[energy['Country Code'].isin(list_of_code)]
energy.fillna(0,inplace=True)
