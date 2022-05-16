import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, dist
from scipy.optimize import curve_fit


def addEdgeToGraph(G, e1, e2, w):
    G.add_edge(e1, e2, weight=w)

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

if __name__ == "__main__":

    # Spatial parameters (in meters)
    delta_space = 75
    max_longitude = 9000
    max_latitude = 12000
    M = ceil(max_longitude/delta_space)
    N = ceil(max_latitude/delta_space)

    # Load graph's data
    file = 'TreeGraph.csv'
    data = pd.read_csv(
        file,
        delim_whitespace=False, header=0,
        names=["node1","node2", "weight","corresp1","corresp2"])

    G = nx.Graph()
    W = np.zeros((len(data)+1,len(data)+1))
    points = []
    dists = []
    weights = []

    # Traverse data and reconstruct the graph's adjacency matrix
    for i in range(len(data)):
        W[data['node1'][i]-1][data['node2'][i]-1] = data['weight'][i]
        W[data['node2'][i]-1][data['node1'][i]-1] = data['weight'][i]
        p1 = ((data['corresp1'][i] - (data['corresp1'][i] % N)) / N * delta_space + delta_space/2, (data['corresp1'][i] % N) * delta_space + delta_space/2)
        p2 = ((data['corresp2'][i] - (data['corresp2'][i] % N)) / N * delta_space + delta_space/2, (data['corresp2'][i] % N) * delta_space + delta_space/2)
        # Create the edges to the graph object
        addEdgeToGraph(G, p1, p2, data['weight'][i])
        if p1 not in points:
            points.append(p1)
        if p2 not in points:
            points.append(p2)
        dists.append(dist(p1,p2))
        weights.append(data['weight'][i])

    # Map the positions of graph nodes to the 2D spatial domain
    pos = {point: point for point in points}
    options = {
        'node_color': 'black',
        'node_size': 15,
        'width': 1,
    }

    # Plot font and labels
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    # Plot graph
    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, **options) # Draw nodes and edges
    plt.axis("on")
    ax.set_xlim(0, max_longitude)
    ax.set_ylim(0, max_latitude)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()

    # Plot edge weights as a function of edge distance
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(dists, weights, s=5)
    plt.xlabel("Edge distance (m)")
    plt.ylabel("Edge weight")
    plt.show()

    # Compute the number of edges as a function of distance
    unique_dists = np.unique(dists)
    print(unique_dists)
    count_dists = []
    for i in unique_dists:
        count = 0
        for j in dists:
            if (j == i):
                count += 1
        count_dists.append(count)
    print(count_dists)

    # Plot the number of edges as a function of edge distance
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(unique_dists, count_dists, s = 15)

    # Perform the fit of a trend line
    p0 = (2000, .1, 50)
    params, cv = curve_fit(monoExp, unique_dists, count_dists, p0)
    m, t, b = params
    plt.plot(unique_dists, monoExp(unique_dists, m, t, b),
        color= 'blue', 
        label="fitted",
        linestyle='--',
        linewidth=1)
    plt.xlabel("Edge distance (m)")
    plt.ylabel("Number of edges")
    plt.show()