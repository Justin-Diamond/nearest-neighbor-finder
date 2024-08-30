import numpy as np
import plotly.graph_objects as go
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import pandas as pd
from graphviz import Digraph
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Read data from CSV file
df = pd.read_csv("frienddata.csv")

# Extract user names
user_names = df["Name"].values

# Drop the 'Name' column to get only the data
data = df.drop("Name", axis=1)

# Replace NaN values with the median value of each column
data = data.fillna(data.median())

# Scale the data for nearest neighbors calculation
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Print out each column with its respective number
print("Columns:")
for i, column in enumerate(data.columns):
    print(f"{i}: {column}")

# Ask the user to type in the numbers of the columns to use for the 3D chart
selected_columns = []
for _ in range(3):
    while True:
        column_number = input("Type the number of a column to use for the 3D chart: ")
        if column_number.isdigit() and int(column_number) in range(len(data.columns)):
            selected_columns.append(data.columns[int(column_number)])
            break
        else:
            print("Invalid input. Please type a valid column number.")

# Select the chosen columns
selected_data = data[selected_columns]

# Convert DataFrame to numpy array
selected_data = selected_data.values


# Function to find the nearest neighbors and their distances
def find_nearest_neighbors(data):
    num_users = data.shape[0]
    nearest_neighbors = np.zeros(num_users, dtype=int)
    nearest_distances = np.zeros(num_users)

    for i in range(num_users):
        min_dist = float("inf")
        nearest_neighbor = -1

        for j in range(num_users):
            if i != j:
                dist = distance.euclidean(data[i], data[j])
                if np.isfinite(dist) and dist < min_dist:  # Check for finite distance
                    min_dist = dist
                    nearest_neighbor = j

        nearest_neighbors[i] = nearest_neighbor
        nearest_distances[i] = min_dist

    return nearest_neighbors, nearest_distances


# Find nearest neighbors using all dimensions
nearest_neighbors_all, nearest_distances_all = find_nearest_neighbors(
    scaled_data.values
)

# Create scatter plot
fig = go.Figure()

# Add points
for i, name in enumerate(user_names):
    fig.add_trace(
        go.Scatter3d(
            x=[selected_data[i, 0]],
            y=[selected_data[i, 1]],
            z=[selected_data[i, 2]],
            mode="markers+text",
            marker=dict(size=8),  # Increase marker size
            name=name,
            text=name,
            textposition="middle right",
            textfont=dict(size=6),  # Increase text size
        )
    )

# Add lines
for i, neighbor in enumerate(nearest_neighbors_all):
    fig.add_trace(
        go.Scatter3d(
            x=[selected_data[i, 0], selected_data[neighbor, 0]],
            y=[selected_data[i, 1], selected_data[neighbor, 1]],
            z=[selected_data[i, 2], selected_data[neighbor, 2]],
            mode="lines",
            line=dict(color="grey", width=2),
            showlegend=False,
        )
    )

# Set layout
fig.update_layout(
    title="3D Scatter Plot of Users with Nearest Neighbors",
    scene=dict(
        xaxis_title=selected_columns[0],
        yaxis_title=selected_columns[1],
        zaxis_title=selected_columns[2],
    ),  # Use selected column names
)

# Show plot
fig.show()

# Create a directed graph
dot = Digraph()

# Add nodes
for name in user_names:
    dot.node(name)

# Add edges with total distance as label
for i, (neighbor, distance) in enumerate(
    zip(nearest_neighbors_all, nearest_distances_all)
):
    dot.edge(user_names[i], user_names[neighbor], label=str(round(distance, 2)))

# Save and render the graph
dot.render("graph_output.gv", view=True)

# Print nearest neighbors in a structured list
print("Nearest Neighbors for Each User (using all dimensions):")
for i, (neighbor, distance) in enumerate(
    zip(nearest_neighbors_all, nearest_distances_all)
):
    print(f"{user_names[i]} -> {user_names[neighbor]} (distance: {distance:.1f})")

# Create a distance matrix for the heatmap using Manhattan distance
dist_matrix = squareform(pdist(scaled_data.values, metric="cityblock"))

# Create a DataFrame for the heatmap
dist_df = pd.DataFrame(dist_matrix, index=user_names, columns=user_names)

# Create a mask to ignore zero values
mask = dist_df == 0

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    dist_df, cmap="coolwarm_r", annot=True, fmt=".1f", mask=mask, cbar_kws={'ticks': [dist_df.min().min(), dist_df.max().max()]}
)  # Use the "coolwarm" colormap and mask zero values
plt.title("Heatmap of Distances Between Users")
plt.show()

# Create 2D scatter plot with linear regression
plt.figure(figsize=(10, 8))
sns.regplot(
    x=selected_data[:, 0],
    y=selected_data[:, 1],
    scatter_kws={"color": "blue"},
    line_kws={"color": "red"},
)
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.title("2D Scatter Plot with Linear Regression")

# Calculate R^2 for the linear regression
model = LinearRegression().fit(selected_data[:, 0].reshape(-1, 1), selected_data[:, 1])
r2 = model.score(selected_data[:, 0].reshape(-1, 1), selected_data[:, 1])
plt.text(0.05, 0.95, f"R^2 = {r2:.1f}", transform=plt.gca().transAxes)

plt.show()
