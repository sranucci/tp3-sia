import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

# Define the weights
weights = [
    [0.6169326813135259, -0.6273719977579983, 0.9660360352747861],
    [0.6169326813135259, -0.6273719977579983, 0.9660360352747861],
    [0.21693268131352583, -0.22737199775799832, 0.5660360352747861],
    [-0.1830673186864742, 0.1726280022420017, 0.16603603527478605]
]

# Create a figure and axis
fig, ax = plt.subplots()

# Set the axis limits
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Create a line object
line, = ax.plot([], [], label='Line', color='blue')


# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,


# Function to update the plot
def update(frame):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    w0, w1, w2 = weights[frame]
    x = np.linspace(-2, 2, 100)
    y = (-w0 - w1 * x) / w2
    ax.plot(x, y, label=f'W{frame} Line', color='blue')
    ax.legend()


# Create the animation
ani = FuncAnimation(fig, update, frames=len(weights), interval=1000)

# Save the animation as a GIF
ani.save('line_animation.gif', writer='pillow', fps=1)

# Display the animation (optional)
plt.show()
