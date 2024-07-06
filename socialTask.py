import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
from networkx.algorithms.community import girvan_newman
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community import louvain_communities
from networkx.algorithms import conductance
from networkx.algorithms.community.quality import modularity

#nodes= pd.read_csv('Nodes.csv')
#edges= pd.read_csv('Edges.csv')

def undirected_G(nodes,edges):
  # Create an empty graph
  G = nx.Graph()

  # Add nodes to the graph with attributes
  for index, row in nodes.iterrows():
        G.add_node(row['ID'], attr1=row['Class'], attr2=row['Gender'])  # Add additional attributes as needed

  # Add edges to the graph
  for index, row in edges.iterrows():
      G.add_edge(row['Source'], row['Target'])

  return G

def directed_G(nodes,edges):
  # Create an empty graph
  G = nx.DiGraph()

  # Add nodes to the graph with attributes
  for index, row in nodes.iterrows():
        G.add_node(row['ID'], attr1=row['Class'], attr2=row['Gender'])  # Add additional attributes as needed

  # Add edges to the graph
  for index, row in edges.iterrows():
      G.add_edge(row['Source'], row['Target'])

  return G

#################################################Evalution_louvain_Graph###################################
def evalution_louvain_G(nodes,edges):

  # Create a directed graph
  G = directed_G(nodes,edges)


  # Define positions for visualization
  pos = nx.spring_layout(G)

  # Detect communities using the Louvain method
  communities_louvain = louvain_communities(G)

  # Convert the communities to labels
  labels_louvain = {node: idx for idx, community in enumerate(communities_louvain) for node in community}

  # Convert the labels to a list
  labels_list_louvain = [labels_louvain[node] for node in G.nodes()]
  
  message = "Louvain Evalution"
  # Calculate modularity
  modularity_score = modularity(G, communities_louvain)
  message += f"\nModularity: {modularity_score:.4f}"

  # Calculate conductance
  conductance_scores = [conductance(G, list(communities_louvain[i])) for i in range(len(communities_louvain))]
  message += f"\nAverage Conductance: {sum(conductance_scores) / len(conductance_scores):.4f}"

  # Calculate Normalized Mutual Information (NMI)

  nodes_df = pd.DataFrame(nodes)
  ground_truth_table = nodes_df.set_index('ID')['Class'].to_dict()
  ground_truth_list = list(ground_truth_table.values())
  nmi_score =  normalized_mutual_info_score(ground_truth_list , labels_list_louvain)
  message += f"\nNormalized Mutual Information (NMI): {nmi_score:.4f}"
  return   message 


# louvain_clustring_directed_G(nodes,edges)

#############################################################Evalution_newman_Graph##################################################

def evalution_newman_G(nodes,edges):

  G=undirected_G(nodes,edges)

  # Define positions for visualization
  pos = nx.spring_layout(G)

  # Detect communities using the Girvan-Newman method
  communities_gen = girvan_newman(G)
  communities = next(communities_gen)  # Get the first partition

  # Convert the communities to labels
  labels_girvan_newman = {node: idx for idx, community in enumerate(communities) for node in community}

  # Convert the labels to a list
  labels_list_girvan_newman = [labels_girvan_newman[node] for node in G.nodes()]
  
  message = "Grivan Newman Evalution"
  # Calculate modularity
  modularity_score = modularity(G, communities)
  message = f"Modularity: {modularity_score:.4f}"

  # Calculate conductance
  conductance_scores = [conductance(G, list(communities[i])) for i in range(len(communities))]
  message += f"\nAverage Conductance: {sum(conductance_scores) / len(conductance_scores):.4f}"

   # Calculate Normalized Mutual Information (NMI)

  nodes_df = pd.DataFrame(nodes)
  ground_truth_table = nodes_df.set_index('ID')['Class'].to_dict()
  ground_truth_list = list(ground_truth_table.values())
  nmi_score =  normalized_mutual_info_score(ground_truth_list , labels_list_girvan_newman)
  message += f"\nNormalized Mutual Information (NMI): {nmi_score:.4f}"
  return   message 

# newman_undirected_G(nodes,edges)
#############################################################basic_undirected_Graph##################################################

def basic_undirected_G(nodes,edges):

  G = G = nx.Graph()   # Use nx.DiGraph() for directed graph
  # Read nodes from CSV file
  nodes_df = nodes
  # Read edges from CSV file
  edges_df = edges

  # Add nodes with attributes
  for _, row in nodes_df.iterrows():
      G.add_node(row['ID'], **row.to_dict())


  # Add edges with attributes
  for _, row in edges_df.iterrows():
      G.add_edge(row['Source'], row['Target'], weight=row.get('weight', 1))



  # Draw the graph
  plt.figure(figsize=(10, 8))
  nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10)

  # Display the plot
  #plt.show()
  pass

# basic_undirected_G(nodes,edges)

#############################################################basic_directed_Graph##################################################

def basic_directed_G(nodes,edges):

  G = G = nx.DiGraph()
  # Read nodes from CSV file
  nodes_df = nodes
  # Read edges from CSV file
  edges_df = edges

  # Add nodes with attributes
  for _, row in nodes_df.iterrows():
      G.add_node(row['ID'], **row.to_dict())


  # Add edges with attributes
  for _, row in edges_df.iterrows():
      G.add_edge(row['Source'], row['Target'], weight=row.get('weight', 1))



  # Draw the graph
  plt.figure(figsize=(10, 8))
  nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10)

  # Display the plot
  #plt.show()
  pass
# basic_directed_G(nodes,edges)

#############################################################link_pagerank##################################################

def link_pagerank_undirected(nodes, edges):

    G = undirected_G(nodes, edges)  # Assuming you have functions for creating undirected and directed graphs


    pagerank_scores = nx.pagerank(G)

    # Step 4: Visualize the network graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Choose a layout for visualization

    # Draw the network graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')

    # Highlight nodes based on PageRank (example)
    node_color = [pagerank_scores[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, alpha=0.8)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='PageRank', ax=plt.gca())  # Specifying the axis for the color bar

    plt.title('Network Visualization using PageRank')

    # Display the plot
    #plt.show()
    pass

# link_pagerank(nodes, edges, 1)

def link_pagerank_directed(nodes, edges):

    G = directed_G(nodes, edges)

    pagerank_scores = nx.pagerank(G)

    # Step 4: Visualize the network graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Choose a layout for visualization

    # Draw the network graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')

    # Highlight nodes based on PageRank (example)
    node_color = [pagerank_scores[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, alpha=0.8)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='PageRank', ax=plt.gca())  # Specifying the axis for the color bar

    plt.title('Network Visualization using PageRank')

    # Display the plot
    #plt.show()
    pass

# link_pagerank(nodes, edges, 1)


#############################################################link_betweenness##################################################

def link_betweenness_undirected(nodes, edges):

    G = undirected_G(nodes, edges)  # Assuming you have functions for creating undirected and directed graphs

    betweenness_scores = nx.betweenness_centrality(G)

    # Step 4: Visualize the network graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Choose a layout for visualization

    # Draw the network graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')

    # Highlight nodes based on Betweenness
    node_color = [betweenness_scores[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, alpha=0.8)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Betweenness', ax=plt.gca())  # Specifying the axis for the color bar

    plt.title('Network Visualization using Betweenness')

    # Display the plot
    #plt.show()
    pass

# link_betweenness_undirected(nodes, edges)
# link_betweenness(nodes, edges, 1)

def link_betweenness_directed(nodes, edges):

    G = directed_G(nodes, edges)

    betweenness_scores = nx.betweenness_centrality(G)

    # Step 4: Visualize the network graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Choose a layout for visualization

    # Draw the network graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')

    # Highlight nodes based on Betweenness
    node_color = [betweenness_scores[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, alpha=0.8)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Betweenness', ax=plt.gca())  # Specifying the axis for the color bar

    plt.title('Network Visualization using Betweenness')

    # Display the plot
    #plt.show()
    pass

# link_betweenness_directed(nodes, edges)


#############################################################link_closeness##################################################

def link_closeness_undirected(nodes, edges):

    G = undirected_G(nodes, edges)  # Assuming you have functions for creating undirected and directed graphs

    closeness_scores = nx.closeness_centrality(G)

    # Step 4: Visualize the network graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Choose a layout for visualization

    # Draw the network graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')

    # Highlight nodes based on Closeness
    node_color = [closeness_scores[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, alpha=0.8)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Closeness', ax=plt.gca())  # Specifying the axis for the color bar

    plt.title('Network Visualization using Closeness')

    # Display the plot
    #plt.show()
    pass

# link_closeness(nodes, edges, 1)

def link_closeness_directed(nodes, edges):

    G = directed_G(nodes, edges)

    closeness_scores = nx.closeness_centrality(G)

    # Step 4: Visualize the network graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Choose a layout for visualization

    # Draw the network graph
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')

    # Highlight nodes based on Closeness
    node_color = [closeness_scores[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, alpha=0.8)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Closeness', ax=plt.gca())  # Specifying the axis for the color bar

    plt.title('Network Visualization using Closeness')

    # Display the plot
    #plt.show()
    pass

import numpy as np

def partition_louvain_clusterid(nodes,edges):

    G=undirected_G(nodes,edges)
    communities = louvain_communities(G)

    # Create a dictionary of nodes for each cluster
    # Create a dictionary of nodes for each cluster
    clusters = {}
    for cluster_id, community in enumerate(communities):
        clusters[cluster_id] = list(community)

    # Separate each partition into its own graph
    cluster_graphs = {}
    for cluster_id, nodes in clusters.items():
        # Create a new graph for each cluster
        subgraph = G.subgraph(nodes).copy()
        cluster_graphs[cluster_id] = subgraph



    # Optional: Plot each cluster (subgraph)
    plt.figure(figsize=(12, 6))
    for i, (cluster_id, subgraph) in enumerate(cluster_graphs.items()):
        plt.subplot(2, len(cluster_graphs) // 2 + 1, i + 1)
        nx.draw(subgraph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title(f"Cluster {cluster_id}")

    plt.tight_layout()
    #plt.show()
    pass

# partition_louvain_clusterid(nodes,edges)


def partition_louvain_directed_clusterid(nodes, edges):
    G = directed_G(nodes, edges)
    communities = louvain_communities(G)

    # Create a dictionary of nodes for each cluster
    # Create a dictionary of nodes for each cluster
    clusters = {}
    for cluster_id, community in enumerate(communities):
        clusters[cluster_id] = list(community)

    # Separate each partition into its own graph
    cluster_graphs = {}
    for cluster_id, nodes in clusters.items():
        # Create a new graph for each cluster
        subgraph = G.subgraph(nodes).copy()
        cluster_graphs[cluster_id] = subgraph

    # Optional: Plot each cluster (subgraph)
    plt.figure(figsize=(12, 6))
    for i, (cluster_id, subgraph) in enumerate(cluster_graphs.items()):
        plt.subplot(2, len(cluster_graphs) // 2 + 1, i + 1)
        nx.draw(subgraph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title(f"Cluster {cluster_id}")

    plt.tight_layout()
    #plt.show()
    pass

# partition_louvain_directed_clusterid(nodes, edges)

def partition_graph_by_degree(nodes, edges, bins=3):
    # Create an undirected graph
    G=undirected_G(nodes,edges)

    # Calculate the degree of each node
    degree = dict(G.degree())

    # Determine bin thresholds
    degree_values = list(degree.values())
    bin_edges = np.histogram_bin_edges(degree_values, bins=bins)

    # Create a dictionary to hold partitions
    partitions = {i: [] for i in range(bins)}

    # Assign nodes to partitions based on their degree
    for node, deg in degree.items():
        bin_index = np.digitize([deg], bin_edges) - 1
        bin_index = bin_index[0] if isinstance(bin_index, np.ndarray) else bin_index  # Handle scalar inputs
        bin_index = min(bin_index, bins - 1)  # Ensure bin index is within valid range
        partitions[bin_index].append(node)

    # Convert partitions to subgraphs
    subgraphs = [G.subgraph(partitions[i]).copy() for i in range(bins)]

    # Create subplots for each partition
    fig, axs = plt.subplots(1, bins, figsize=(16, 6))

    # Visualize each partition
    for i, subgraph in enumerate(subgraphs):
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, ax=axs[i])
        axs[i].set_title(f"Partition {i + 1}\nDegree bin: {bin_edges[i]} to {bin_edges[i + 1]}")

    plt.tight_layout()
    #plt.show()

    return subgraphs
    pass

# Usage example:
# partition_graph_by_degree(nodes, edges, bins=6)

def partition_graph_by_degree_directed(nodes, edges, bins=3):
    # Create an undirected graph
    G=directed_G(nodes,edges)

    # Calculate the degree of each node
    degree = dict(G.degree())

    # Determine bin thresholds
    degree_values = list(degree.values())
    bin_edges = np.histogram_bin_edges(degree_values, bins=bins)

    # Create a dictionary to hold partitions
    partitions = {i: [] for i in range(bins)}

    # Assign nodes to partitions based on their degree
    for node, deg in degree.items():
        bin_index = np.digitize([deg], bin_edges) - 1
        bin_index = bin_index[0] if isinstance(bin_index, np.ndarray) else bin_index  # Handle scalar inputs
        bin_index = min(bin_index, bins - 1)  # Ensure bin index is within valid range
        partitions[bin_index].append(node)

    # Convert partitions to subgraphs
    subgraphs = [G.subgraph(partitions[i]).copy() for i in range(bins)]

    # Create subplots for each partition
    fig, axs = plt.subplots(1, bins, figsize=(16, 6))

    # Visualize each partition
    for i, subgraph in enumerate(subgraphs):
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, ax=axs[i])
        axs[i].set_title(f"Partition {i + 1}\nDegree bin: {bin_edges[i]} to {bin_edges[i + 1]}")

    plt.tight_layout()
    #plt.show()

    return subgraphs
    pass
# Usage example:
# partition_graph_by_degree_directed(nodes, edges, bins=6)

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity
import numpy as np


def newman_partitioning_by_degree_bins(nodes, edges, num_bins=3):
    """
    Partition the graph based on Girvan-Newman method and group nodes within each community by degree bins.

    Args:
        nodes (pd.DataFrame): DataFrame containing nodes information with 'ID' column.
        edges (pd.DataFrame): DataFrame containing edges information with 'Source' and 'Target' columns.
        num_bins (int): Number of bins to create for node degree.

    Returns:
        None
    """
    # Create an undirected graph
    G =undirected_G(nodes,edges)

    # Run the Girvan-Newman algorithm to detect communities
    community_gen = girvan_newman(G)
    communities = next(community_gen)  # Get the first partition

    # Convert communities to dictionary form
    community_dict = {node: idx for idx, community in enumerate(communities) for node in community}

    # Calculate node degrees
    degree_dict = dict(G.degree())

    # Define degree bins
    degree_values = list(degree_dict.values())
    bins = np.histogram_bin_edges(degree_values, bins=num_bins)

    # Function to assign a degree bin
    def assign_degree_bin(degree):
        for i in range(len(bins) - 1):
            if bins[i] <= degree < bins[i + 1]:
                return i
        return len(bins) - 2

    # Group nodes within each community by their degree bins
    community_partitions_by_degree_bins = {}
    for idx in range(len(communities)):
        community_partitions_by_degree_bins[idx] = {}

    for node in G.nodes():
        community_idx = community_dict[node]
        degree = degree_dict[node]
        degree_bin = assign_degree_bin(degree)
        if degree_bin not in community_partitions_by_degree_bins[community_idx]:
            community_partitions_by_degree_bins[community_idx][degree_bin] = []
        community_partitions_by_degree_bins[community_idx][degree_bin].append(node)

    # Plot each community partition based on degree bins in the same screen
    # Calculate the total number of plots
    total_plots = sum(len(partition) for partition in community_partitions_by_degree_bins.values())

    # Create subplots grid
    num_rows = int(np.ceil(total_plots / 3))
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    plot_index = 0
    for idx, partition_by_degree_bins in community_partitions_by_degree_bins.items():
        for degree_bin, nodes_list in partition_by_degree_bins.items():
            # Calculate the row and column index for the current subplot
            row_index = plot_index // 3
            col_index = plot_index % 3

            # Create a subgraph with nodes of specific degree bin within a community
            subgraph = G.subgraph(nodes_list)

            # Plot the subgraph in the corresponding subplot
            ax = axes[row_index, col_index]
            pos = nx.spring_layout(subgraph)
            nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, ax=ax)
            ax.set_title(f"Community {idx + 1}, Degree Bin {degree_bin}")

            plot_index += 1

    # Adjust layout and display the plots
    plt.tight_layout()
    #plt.show()
    pass

def newman_partitioning_by_degree_bins_directed(nodes, edges, num_bins=3):
    """
    Partition the graph based on Girvan-Newman method and group nodes within each community by degree bins.

    Args:
        nodes (pd.DataFrame): DataFrame containing nodes information with 'ID' column.
        edges (pd.DataFrame): DataFrame containing edges information with 'Source' and 'Target' columns.
        num_bins (int): Number of bins to create for node degree.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    # Create an undirected graph
    G = directed_G(nodes, edges)

    # Run the Girvan-Newman algorithm to detect communities
    community_gen = girvan_newman(G)
    communities = next(community_gen)  # Get the first partition

    # Convert communities to dictionary form
    community_dict = {node: idx for idx, community in enumerate(communities) for node in community}

    # Calculate node degrees
    degree_dict = dict(G.degree())

    # Define degree bins
    degree_values = list(degree_dict.values())
    bins = np.histogram_bin_edges(degree_values, bins=num_bins)

    # Function to assign a degree bin
    def assign_degree_bin(degree):
        for i in range(len(bins) - 1):
            if bins[i] <= degree < bins[i + 1]:
                return i
        return len(bins) - 2

    # Group nodes within each community by their degree bins
    community_partitions_by_degree_bins = {}
    for idx in range(len(communities)):
        community_partitions_by_degree_bins[idx] = {}

    for node in G.nodes():
        community_idx = community_dict[node]
        degree = degree_dict[node]
        degree_bin = assign_degree_bin(degree)
        if degree_bin not in community_partitions_by_degree_bins[community_idx]:
            community_partitions_by_degree_bins[community_idx][degree_bin] = []
        community_partitions_by_degree_bins[community_idx][degree_bin].append(node)

    # Plot each community partition based on degree bins in the same screen
    num_cols = min(num_bins, 3)  # Maximum of 3 columns
    num_rows = len(communities)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    for idx, partition_by_degree_bins in community_partitions_by_degree_bins.items():
        for degree_bin, nodes_list in partition_by_degree_bins.items():
            # Create a subgraph with nodes of specific degree bin within a community
            subgraph = G.subgraph(nodes_list)

            # Plot the subgraph
            row_index = idx
            col_index = degree_bin if num_bins <= 3 else degree_bin % 3
            ax = axes[row_index, col_index] if num_rows > 1 else axes[col_index]
            pos = nx.spring_layout(subgraph)
            nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, ax=ax)
            ax.set_title(f"Community {idx + 1}, Degree Bin {degree_bin}")

    # Adjust layout and display the plot
    plt.tight_layout()
    #plt.show()
    pass



# Example usage:
# Assuming 'nodes' and 'edges' are pandas DataFrames containing the node and edge data.
# newman_partitioning_by_degree_bins_directed(nodes, edges, num_bins=3)


import networkx as nx
import matplotlib.pyplot as plt

def girvan_newman_algorithm(graph):
    """Finds communities in a graph using the Girvan-Newman algorithm."""
    # Copy the original graph to avoid modifying it directly
    g = graph.copy()
    # Initialize a list to hold the detected communities
    communities = []

    while g.number_of_edges() > 0:
        # Calculate the betweenness centrality for all edges in the graph
        betweenness = nx.edge_betweenness_centrality(g)

        # Find the edge with the highest betweenness centrality
        max_betweenness_edge = max(betweenness, key=betweenness.get)

        # Remove the edge with the highest betweenness centrality
        g.remove_edge(*max_betweenness_edge)

        # Check if the graph has become disconnected (separated into components)
        connected_components = list(nx.connected_components(g))

        # If there are multiple connected components, we have found communities
        if len(connected_components) > 1:
            # Add the detected communities to the list
            communities.extend(connected_components)
            # Exit the loop as the graph is now partitioned into communities
            break

    # Return the detected communities
    return communities
    pass

# Example usage
def girvan_newman_algoritm_undirected(nodes,edges):

    # Create a sample graph
    G =undirected_G(nodes,edges)

    # Run the Girvan-Newman algorithm
    communities = girvan_newman_algorithm(G)

    # Determine the number of communities
    num_communities = len(communities)

    # Create subplots: (num_communities rows, 1 column)
    fig, axes = plt.subplots(num_communities, 1, figsize=(8, num_communities * 6))

    # Plot each community in a separate subplot
    for i, community in enumerate(communities):
        # Create a subgraph for each community
        subgraph = G.subgraph(community)

        # Plot the subgraph in the i-th subplot
        nx.draw(subgraph, ax=axes[i], with_labels=True, node_color='lightblue', edge_color='gray')
        axes[i].set_title(f"Community {i + 1}")

    # Show all subplots in the same figure
    plt.tight_layout()
    #plt.show()
    pass

# girvan_newman_algoritm_undirected(nodes,edges)

# def girvan_newman_algoritm_directed(nodes, edges):
#     # Create a sample graph
#     G = directed_G(nodes, edges)
#
#     # Run the Girvan-Newman algorithm
#     communities = girvan_newman_algorithm(G)
#
#     # Determine the number of communities
#     num_communities = len(communities)
#
#     # Create subplots: (num_communities rows, 1 column)
#     fig, axes = plt.subplots(num_communities, 1, figsize=(8, num_communities * 6))
#
#     # Plot each community in a separate subplot
#     for i, community in enumerate(communities):
#         # Create a subgraph for each community
#         subgraph = G.subgraph(community)
#
#         # Plot the subgraph in the i-th subplot
#         nx.draw(subgraph, ax=axes[i], with_labels=True, node_color='lightblue', edge_color='gray')
#         axes[i].set_title(f"Community {i + 1}")
#
#     # Show all subplots in the same figure
#     plt.tight_layout()
#     plt.show()

# girvan_newman_algoritm_directed(nodes, edges)
# link_closeness(nodes, edges, 1)

