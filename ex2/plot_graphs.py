import json
import os
import pandas as pd

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def MSE_graph():
    path = "./results"
    files = os.listdir(path)
    files.sort()
    file_name = f"{path}/{files[0]}"

    with open(file_name, "r") as file:
        data = json.load(file)
        error_values = data["error"]


    error_array = np.array(error_values[1:])


    x_values = np.arange(1, len(error_array) + 1)

    fig = go.Figure(data=go.Scatter(x=x_values, y=error_array, mode='lines', name='Error'))
    fig.update_layout(title="Error Line Graph", xaxis_title="Epoch", yaxis_title="Error Value")

    print(min(error_array))
    fig.show()


def linear_training():
    # Read the data from the CSV file
    data = pd.read_csv('./results/test_lineal.csv', header=None, names=['X', 'Y'])

    # Create a scatter plot
    fig = px.scatter(data, x='X', y='Y', title='')
    fig.update_xaxes(title_text='Predicted')
    fig.update_yaxes(title_text='Expected')
    line_trace = go.Scatter(x=[min(data['X']), max(data['X'])], y=[min(data['X']), max(data['X'])], mode='lines',
                            name='y=x', line=dict(color='green'))
    fig.add_trace(line_trace)   # Show the plot
    # Calculate the mean of Y values


    fig.show()

def error_test():
    list = [68.70750660281355,
1036.884169461413,
422.0052712474584,
405.5805049432101,
218.3986431659568,
66.51507664083184,
18.534899009162718,
922.3672326240386,
679.3400463426531,
1109.6418488738354,
32.11214402675747,
531.1777634856331,
35.964474675223535,
29.56140692423835,
381.0973594738524,
344.8168524186842,
332.7584292981121,
438.5464663799502,
910.0213583427516,
430.7975629293738,
650.9755876033788,
21.100809817557433,
8.693781280809867,
33.09166348646815,
15.076899086908439
            ]
    sum = 0
    for elem in list:
        sum += elem

    print(sum/len(list))

MSE_graph()