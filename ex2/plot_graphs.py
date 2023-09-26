import json
import os

import numpy as np
import plotly.graph_objects as go


path = "./results"
files = os.listdir(path)
files.sort()
file_name = f"{path}/{files[-1]}"

with open(file_name, "r") as file:
    data = json.load(file)
    error_values = data["error"]


error_array = np.array(error_values[1:])


x_values = np.arange(1, len(error_array) + 1)

fig = go.Figure(data=go.Scatter(x=x_values, y=error_array, mode='lines', name='Error'))
fig.update_layout(title="Error Line Graph", xaxis_title="Epoch", yaxis_title="Error Value")


fig.show()


