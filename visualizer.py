import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Paths to the saved models
high_acc_model_path = "high_acc_model.h5"
low_acc_model_path = "low_acc_model.h5"

# Load the models
high_acc_model = tf.keras.models.load_model(high_acc_model_path)
low_acc_model = tf.keras.models.load_model(low_acc_model_path)

def random_color():
    return np.random.rand(3)

# Function to draw a cuboid-like structure with faces and edges
def hollow(ax, x, y, z, x_size, y_size, z_size, color='skyblue', linewidth=1, alpha=1, label=None,padding_top_hover=10):
    # Create 8 vertices for the cuboid
    x_pos, y_pos, z_pos = x, y, z
    # Vertices of the cuboid
    vertices = np.array([
        [x_pos, y_pos, z_pos], 
        [x_pos + x_size, y_pos, z_pos], 
        [x_pos + x_size, y_pos + y_size, z_pos],
        [x_pos, y_pos + y_size, z_pos],
        [x_pos, y_pos, z_pos + z_size],
        [x_pos + x_size, y_pos, z_pos + z_size],
        [x_pos + x_size, y_pos + y_size, z_pos + z_size],
        [x_pos, y_pos + y_size, z_pos + z_size]
    ])
    
    # List of edges of the cuboid (pairs of vertices)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4], # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical edges
    ]
    
    # Plot edges of the cuboid
    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color=color, linewidth=linewidth, alpha=alpha)

    # Faces of the cuboid (each face is a quadrilateral)
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left face
    ]
    
    # Create Poly3DCollection for the faces and add them to the plot
    poly3d = Poly3DCollection(faces, color=color, alpha=alpha,linewidth=linewidth)
    ax.add_collection3d(poly3d)
    
    if label:
    # Offset the position above the cuboid
        label_position = [x_pos + x_size / 2, y_pos + y_size / 2, z_pos + z_size + padding_top_hover]
        
        # Apply a simulated diagonal rotation by adjusting the position slightly
        # Adjust the position slightly in the X and Y axes for a "diagonal" effect
        offset_x = 0.1
        offset_y = 0.1
        label_position[0]*=1+ offset_x
        label_position[1]*=1+offset_y
        # Use ax.text to position the label in 3D with the desired rotation effect
        ax.text(
            label_position[0], label_position[1], label_position[2],
            label,
            color='black',
            fontsize=12,
            ha='center',
            va='center',
        )

    return ax


# Function to visualize the model layers as cubes in 3D
def visualize_model_layers_as_cubes(model, title,scale = 0.1,gap=5,layergap=100,length_cube = 10):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    layer_names=[]
    layer_shapes = []

    _maxx=0
    for i, layer in enumerate(model.layers):
        layer_names.append(layer.name)
        if hasattr(layer, 'output'):
            layer_shape = layer.output.shape
            # Define the size of the cube based on the shape of the layer
            _maxx = max(_maxx,max([i for i in layer_shape if i is not None]))
            layer_shapes.append(layer_shape)
        else:
            layer_shapes.append("N/A")

    # Create cubes for each layer
    nextX=0
    _maxx*=scale*1.1
    for i, shape in enumerate(layer_shapes):
        if shape != "N/A":

            startCuboidX = nextX
            nextX+=scale*gap*2
            # If output shape exists, use it to determine the cube's scale
            if isinstance(shape, tuple):
                # Create a cube with fixed dimensions based on the layer's output shape
                x_size, y_size, z_size = scale,scale,scale # Default cube size
                print(shape)
                for dim in shape:
                    if dim is None:continue
                    nextX+=scale*(gap+length_cube)
                    x_size =scale*length_cube
                    y_size =scale*dim 
                    z_size =scale*dim
                    # Create a cube using a scatter plot
                    ax.bar3d(nextX, -.5*y_size, -.5*z_size,
                            x_size, y_size, z_size, color=random_color(), linewidth=100*scale, alpha=0.5)
                    ax.bar3d(nextX+scale, -0.5*scale, -0.5*scale,
                             scale*(gap-length_cube),scale,scale,color='red', linewidth=1*scale)#seperator
            endCuboidX = nextX + scale*gap*2
            nextX+=scale*(gap+layergap)

            # Add the cube to the plot
            hollow(ax,startCuboidX, -.5*_maxx, -.5*_maxx,
                      endCuboidX - startCuboidX, _maxx, _maxx,color=random_color(), linewidth=scale,alpha=.1,label=layer_names[i])
            ax.bar3d(endCuboidX, -.5*scale, -.5*scale,
                     scale*(layergap), scale, scale, color='red', linewidth=1*scale)
            
    # Set labels and title
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Y (constant)")
    ax.set_zlabel("Z (depth)")
    
    # Adjust plot limits
    # ax.set_xlim([min(x_positions) - 1, max(x_positions) + 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    
    plt.show()

# Visualize high accuracy model
visualize_model_layers_as_cubes(high_acc_model, "High Accuracy Model")

# Visualize low accuracy model
visualize_model_layers_as_cubes(low_acc_model, "Low Accuracy Model")
