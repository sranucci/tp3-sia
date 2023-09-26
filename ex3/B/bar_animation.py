import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import csv

with open("./results.csv","r") as f:
    reader = csv.reader(f)
    data_x = []
    data_y = []
    for x,y in reader:
        data_y.append(float(y))
        data_x.append(float(x))

# Your y value - replace this with your desired y value
y = 0

# Create a figure and axis
fig, ax = plt.subplots()

# Set the initial data with a single centered bar
bar = ax.bar(0, y, align='center')

# Function to update the bar chart for each frame
def update(frame):
    # Randomly generate new y value for demonstration purposes
    new_y = data_y[frame]
    new_x = data_x[frame]
    ax.set_title(f"alpha : {new_x}\n error: {new_y}")
    # Update the height of the bar
    bar[0].set_height(new_y)

# Create the animation
ani = FuncAnimation(fig, update, frames=range(100), repeat=False, interval=2000)  # Update every 2 seconds

# Customize the plot (labels, titles, etc.)
ax.set_ylabel('error')
ax.set_title('Animated Centered Bar Chart Example')
# Hide the x-axis labels and ticks
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylim(min(data_y),max(data_y))

ani.save("./error_comparison.gif",writer="pillow",fps=2)

# Show the animated chart
plt.show()
