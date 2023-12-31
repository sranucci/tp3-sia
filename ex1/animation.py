import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
import json


def animate_lines(input_data, gif_name, file_name=None):
    if file_name is None:
        path = "./results"
        files = os.listdir(path)
        files.sort()
        file_name = f"{path}/{files[-1]}"

    with open(file_name) as file:
        results = json.load(file)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the axis limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Create a line object
    line, = ax.plot([], [], label='Line', color='blue')

    # Create a scatter plot for input coordinates
    input_data = np.array(input_data)
    scatter = ax.scatter([], [], c='red', marker='o', label='Input Coordinates')

    # Function to initialize the plot
    def init():
        line.set_data([], [])
        return line,

    # Function to update the plot
    def update(frame):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        w0, w1, w2 = results['weights'][frame]
        x = np.linspace(-2, 2, 100)
        y = (-w0 - w1 * x) / w2
        line.set_data(x, y)  # Update the line

        # Update the input coordinates
        scatter.set_offsets(input_data)

        # Update the legend with the line number
        legend_label = f'Line {frame + 1} (W0={w0:.2f}, W1={w1:.2f}, W2={w2:.2f})'
        ax.legend([legend_label])

        # Add labels next to each point
        for i, (xi, yi) in enumerate(input_data):
            ax.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(10, 10), fontsize=8)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(results['weights']), init_func=init, interval=1000)

    # Save the animation as a GIF
    ani.save(f'./gifs/{gif_name}.gif', writer='pillow', fps=2)

    # Display the animation (optional)
    plt.show()

