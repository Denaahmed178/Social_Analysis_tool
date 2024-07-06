# -*- coding: utf-8 -*-
"""
Created on Sun May  5 04:58:40 2024

@author: green
"""
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import pandas as pd
import networkx as nx
# import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import socialFunctions

from tkinter import *
from tkinter.ttk import Separator

import pandas as pd
import socialFunctions
import socialTask

"""  GUI  """
def create_gui():
    graph = None
    root = tk.Tk()
    plt.figure()
    def create_popup(G):
      popup = tk.Toplevel(root)
      popup.title("Popup Window")
      show_graph_in_canvas(G, popup)
      # Bind function to execute when window is closed
      #popup.protocol("WM_DELETE_WINDOW", on_popup_close) 
        # Update GUI
      #popup.mainloop()
    def on_popup_close():
        plt.gca().clear()
    def browse_file(entry_widget):
        filename = filedialog.askopenfilename()
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)

    checkbox_directed_var = tk.BooleanVar()
    def create_graph():
        try:
            node_df_path = entry_node_df_path.get()
            edge_df_path = entry_edge_df_path.get()
            directed = checkbox_directed_var.get()

            G, _, _ = socialFunctions.create_graph_from_csv(node_df_path, edge_df_path, directed)   
            set_graph(G)
                        # Visualize graph
            plt.figure(figsize=(8, 6))
                        # Example layout, you can choose other layouts
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='black', linewidths=1, font_size=10)
            plt.title("Graph Visualization")
            plt.axis('off')
            # Embedding matplotlib figure in Tkinter window
            #show_graph_in_canvas(G, root)
            create_popup(G)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    def get_graph():
        # Retrieve the graph using the create_graph function
        global graph
        return graph
    
    def set_graph(G):
          global graph
          graph=G
    def show_graph_in_canvas(G, root):
         # Clear the canvas before display the new graph 
         #plt.clf()
         # Embedding matplotlib figure in Tkinter window
         #root.
         canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
         canvas.draw()
         canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
         # Update GUI
         root.update_idletasks()
    def show_output_in_new_window(output_text):
        new_window = tk.Toplevel()
        new_window.title("Output Window")
        output_label = ttk.Label(new_window, text=output_text)
        output_label.pack()
        new_window.mainloop()

    """ GUI task-1 change nodes and edges attributes """
    def update_graph():
            try:
                node_color = entry_node_color.get()
                node_size = int(entry_node_size.get())
                edge_color = entry_edge_color.get()
                # Handling the case where entry_edge_width is empty
                edge_width_str = entry_edge_width.get()
                edge_width = int(edge_width_str) if edge_width_str else 1
    
                #G = create_graph()
                G= get_graph()
                edge_colors, edge_widths =socialFunctions.set_edge_attributes(G, edge_color, edge_width)
                node_sizes, node_colors = socialFunctions.set_new_attributes(G, node_color, node_size)
                plt.clf()
                plt.figure(figsize=(8, 6))
                pos = nx.spring_layout(G)
                # Drawing nodes and edges
                nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
                nx.draw_networkx_labels(G, pos)
                set_graph(G)
                # Embedding matplotlib figure in Tkinter window
                #show_graph_in_canvas(G, root)
                create_popup(G)
    
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
    """ GUI task-1 display colored graph & histogram chart """
    def execute_create_colored_graph():############################################
            node_df_path = entry_node_df_path.get()
            edge_df_path = entry_edge_df_path.get()
            directed = checkbox_directed_var.get()
    
            try:
            # Load node and edge dataframes
                 node_df = pd.read_csv(node_df_path)
                 edge_df = pd.read_csv(edge_df_path)
    
            # Create colored graph
                 G, class_colors = socialFunctions.create_colored_graph(node_df, edge_df, directed)
                 plt.clf()
                 # Plot the colored graph
                 fig, ax = plt.subplots(figsize=(8, 6))
                 pos = nx.spring_layout(G)  # Example layout, you can choose other layouts
                 nx.draw(G, pos, ax=ax, with_labels=True, node_color=[data['color'] for node, data in G.nodes(data=True)], node_size=500, edge_color='black', linewidths=1, font_size=10)
                 ax.set_title("Colored Graph")
                 ax.set_axis_off()
                 # def draw_colored_graph(G, directed=True):
                 #     pos = nx.spring_layout(G)  # Define the layout for better visualization
                 #     node_colors = [data['color'] for node, data in G.nodes(data=True)]
    
        # Embedding matplotlib figure in Tkinter window
                 #show_graph_in_canvas(G, root)
                 create_popup(G)
            # Plot the colored graph
                 plt.figure(figsize=(8, 6))
                 
                 socialFunctions.draw_colored_graph(G, directed)
    
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
    def execute_show_histogram():
                if display_histogram_var.get():
                    try:
                        # Load node and edge dataframes
                        node_df = pd.read_csv(entry_node_df_path.get())
                        edge_df = pd.read_csv(entry_edge_df_path.get())
            
                        # Calculate class counts and colors
                        unique_classes = node_df['Class'].unique()
                        class_counts = node_df['Class'].value_counts()
                        class_colors = dict(zip(unique_classes, plt.cm.tab10.colors[:len(unique_classes)]))
                        # Plot class histogram
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plt.bar(range(len(class_counts)), class_counts.values, color=[
                            class_colors.get(cls, 'gray') for cls in class_counts.index])
                        plt.xticks(range(len(class_counts)), class_counts.index)
                        plt.xlabel('Class')
                        plt.ylabel('Count')
                        plt.title('Class Distribution')
                        
                        popup = tk.Toplevel(root)
                        popup.title("Popup Window")
            
                        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
                        canvas.draw()  
                        canvas.get_tk_widget().grid(row=0, column=0, columnspan=4, padx=4, pady=4)
                    except Exception as e:
                        messagebox.showerror("Error", f"An error occurred: {str(e)}")
                else:
                    # Clear the canvas if metrics are not to be displayed
                    plt.clf()
    """ GUI task-2 change layout type """
    def change_layout(value):
              #  G = create_graph()
                plt.clf()
                G=get_graph()
                socialFunctions.visualize_graph_with_layout(G, layout_algorithm=value)
                set_graph(G)
                #show_graph_in_canvas(G, root)
                create_popup(G)
            
    """ GUI task-3 compute matrics for the graph """
    def toggle_graph_metrics():#############################################
        if display_metrics_var.get():
            try:
                node_df_path = entry_node_df_path.get()  # Retrieve the path from the entry widget
                edge_df_path = entry_edge_df_path.get()  # Retrieve the path from the entry widget
                directed = checkbox_directed_var.get()
    
                # Ensure the function is called with the correct arguments
                G, _, _ = socialFunctions.create_graph_from_csv(node_df_path, edge_df_path, directed)
                metrics = socialFunctions.compute_graph_metrics(G, directed=directed)
                # Convert metrics to a matrix
                matrix_data = np.random.rand(5, 5)  # Replace this with your matrix data
                plt.imshow(matrix_data, cmap='viridis')
                plt.colorbar()
                
                popup = tk.Toplevel(root)
                popup.title("Popup Window")
                
                # Embed the matrix graph into Tkinter window
                canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
                canvas.draw()
                canvas.get_tk_widget().grid(row=0, column=0, columnspan=4, padx=4, pady=4)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            # Clear the canvas if metrics are not to be displayed
            plt.clf()
    def create_graph_metrics(G):
       G= get_graph()
       plt.clf()
       socialFunctions.compute_graph_metrics(G)
       
    def create_graph_statistics():
        directed = checkbox_directed_var.get()
        G = get_graph()
        output_text = socialFunctions.compute_graph_statistics(G, directed)
        show_output_in_new_window(output_text)

    """ GUI task-4 filter graph """
    """ #1 Filtering nodes based on centrality measures """
    def filter_by_centrality():
            G= get_graph()
            plt.clf()
            selected_option = dropdown_centrality.cget('text')
            thershold_value = float(entry_centrality_threshold.get())
            socialFunctions.filter_nodes_by_centrality(G, selected_option,thershold_value)
            #show_graph_in_canvas(G, root)
            create_popup(G)
    """ #2 Filtering nodes based on their membership """
    def filter_by_membership(membership):
        G= get_graph()
        plt.clf()
        community_nodes =socialFunctions.filter_nodes(G, community=membership)
        G_community_filtered = socialFunctions.filter_nodes_by_community(G, community_nodes)
        nx.draw(G_community_filtered, with_labels=True, node_color='skyblue')
        #show_graph_in_canvas(G, root)
        create_popup(G)
    """ Task 5  compare algorithms """
    def compare_community_algorithms():
        G = get_graph()
        plt.clf()
        msg_modularity, msg_louvain = socialFunctions.compare_community_detection_algorithms(G)
        output_text = f"Modularity:\n{msg_modularity}\n\nLouvain:\n{msg_louvain}"
        show_output_in_new_window(output_text)


    #nodes= pd.read_csv(r'D:\4th year\Second smester\Social Media\data\Node.csv')
    #edges= pd.read_csv(r'D:\4th year\Second smester\Social Media\data\Edges.csv')
    def get_nodes_edges():
        nodes = entry_node_df_path.get()
        edges = entry_edge_df_path.get()
        nodes= pd.read_csv(nodes)
        edges= pd.read_csv(edges)
        return nodes, edges
    def execute_louvain_evalution():
        nodes, edges = get_nodes_edges()
        evalution_msg  = socialTask.evalution_louvain_G(nodes=nodes,edges=edges)
        show_output_in_new_window(evalution_msg)
    def execute_newman_evalution():
        nodes, edges = get_nodes_edges()
        evalution_msg = socialTask.evalution_newman_G(nodes=nodes,edges=edges)
        show_output_in_new_window(evalution_msg)
    def execute_louvain_directed():
        nodes, edges = get_nodes_edges()
        socialTask.partition_louvain_directed_clusterid(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("Louvain on Directed graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
        
    def execute_louvain_undirected():
        nodes, edges = get_nodes_edges()
        socialTask.partition_louvain_clusterid(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("Louvain on unDirected graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
        
    def execute_partioning_by_degree_undirected():
        nodes, edges = get_nodes_edges()
        socialTask.partition_graph_by_degree(nodes=nodes,edges=edges,bins=6)
        popup = tk.Toplevel(root)
        popup.title("Clusting on unDirected graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
        
    def execute_partioning_by_degree_directed():
        nodes, edges = get_nodes_edges()
        socialTask.partition_graph_by_degree_directed(nodes=nodes,edges=edges,bins=6)
        popup = tk.Toplevel(root)
        popup.title("Clustering on Directed graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
        
    def execute_patrtioning_by_newman_and_degree():
        nodes, edges = get_nodes_edges()
        socialTask.newman_partitioning_by_degree_bins(nodes=nodes,edges=edges,num_bins=4)
        popup = tk.Toplevel(root)
        popup.title("Cluster by Newman and degree")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    def execute_patrtioning_by_newman_and_degree_directed():
        nodes, edges = get_nodes_edges()
        socialTask.newman_partitioning_by_degree_bins_directed(nodes=nodes,edges=edges,num_bins=4)
        popup = tk.Toplevel(root)
        popup.title("Cluster by Newman and degree Directed graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
        
    def execute_newman_undirected():
        nodes, edges = get_nodes_edges()
        socialTask.girvan_newman_algoritm_undirected(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("Newman on Directed graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
        
    def execute_basic_undirected():
        nodes, edges = get_nodes_edges()
        socialTask.basic_undirected_G(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("Basic Undirected graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
        
    def execute_basic_directed():
        nodes, edges = get_nodes_edges()
        socialTask.basic_directed_G(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("Basic Undirected graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    def execute_link_pagerank_undirected():
        nodes, edges = get_nodes_edges()
        socialTask.link_pagerank_undirected(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("Page Rank unDirected graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    def execute_link_pagerank_directed():
        nodes, edges = get_nodes_edges()
        socialTask.link_pagerank_directed(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("Page Rank Directed graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    def execute_link_betweenness_undirected():
        nodes, edges = get_nodes_edges()
        socialTask.link_betweenness_undirected(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("link Betweenness unDirected graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    def execute_link_betweenness_directed():
        nodes, edges = get_nodes_edges()
        socialTask.link_betweenness_directed(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("link Betweenness Directed graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    def execute_link_closeness_undirected():
        nodes, edges = get_nodes_edges()
        socialTask.link_closeness_undirected(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("link Closeness unDirected graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    def execute_link_closeness_directed():
        nodes, edges = get_nodes_edges()
        socialTask.link_closeness_directed(nodes=nodes, edges=edges)
        popup = tk.Toplevel(root)
        popup.title("link Closeness Directed graph")
        canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=5)
    # def run_analysis():
    #     try:
    #         node_df_path = entry_node_df_path.get()
    #         edge_df_path = entry_edge_df_path.get()
    #         directed = checkbox_directed_var.get()

    #         G, _, _ = create_graph_from_csv(node_df_path, edge_df_path, directed)

    #         # Visualize graph
    #         plt.figure(figsize=(8, 6))
    #         # Example layout, you can choose other layouts
    #         pos = nx.spring_layout(G)
    #         nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='black', linewidths=1, font_size=10)
    #         plt.title("Graph Visualization")
    #         plt.axis('off')

    #         # Embedding matplotlib figure in Tkinter window
    #         show_graph_in_canvas(G, root)

    #         compute_graph_metrics(G, directed=False)
    #         compare_community_detection_algorithms(G)

    #         messagebox.showinfo("Analysis Complete", "Graph analysis completed successfully.")

    #     except Exception as e:
    #         messagebox.showerror("Error", f"An error occurred: {str(e)}")

        
  
    root.title("Graph Analysis Tool")

# Node DataFrame Path
    label_node_df = ttk.Label(root, text="Node DataFrame Path:", foreground="blue", font=("Arial", 11, "bold"))
    label_node_df.grid(row=0, column=0)
    entry_node_df_path = ttk.Entry(root, width=30)
    entry_node_df_path.grid(row=0, column=1,  padx=1, columnspan=1, pady=1)
    button_browse_node_df = ttk.Button(root, text="Browse", command=lambda: browse_file(entry_node_df_path))
    button_browse_node_df.grid(row=0, column=2,  padx=2,  pady=2)
    
    
    label_edge_df = ttk.Label(root, text="Edge DataFrame Path:", foreground="blue", font=("Arial", 11, "bold"))
    label_edge_df.grid(row=1, column=0)
    entry_edge_df_path = ttk.Entry(root, width=30)
    entry_edge_df_path.grid(row=1, column=1,  padx=2,  pady=2)
    button_browse_edge_df = ttk.Button(root, text="Browse", command=lambda: browse_file(entry_edge_df_path))
    button_browse_edge_df.grid(row=1, column=2,  padx=2,  pady=2)
    
    label_centrality_threshold = ttk.Label(root, text="Directed:",foreground="blue", font=("Arial", 11, "bold"))
    label_centrality_threshold.grid(row=0, column=3,  padx=2,  pady=2)
    checkbox_directed = ttk.Checkbutton(root, text="Directed Graph", variable=checkbox_directed_var)
    checkbox_directed.grid(row=0, column=4, padx=2, pady=2, columnspan=1)
    
    call_graph = ttk.Button(root, text="Create Graph", command=create_graph)
    call_graph.grid(row=1, column=3,  padx=2,  pady=2)
    
  
    label_node_color = ttk.Label(root, text="Node Color:", foreground="blue", font=("Arial", 11, "bold"))
    label_node_color.grid(row=3, column=0)
    entry_node_color = ttk.Entry(root, width=20)
    entry_node_color.grid(row=3, column=1,  padx=2,  pady=2)
    
    label_node_size = ttk.Label(root, text="Node Size:",foreground="blue", font=("Arial", 11, "bold"))
    label_node_size.grid(row=3, column=2)
    entry_node_size = ttk.Entry(root, width=10)
    entry_node_size.grid(row=3, column=3,  padx=2,  pady=2)
    
    label_edge_color = ttk.Label(root, text="Edge Color:",foreground="blue", font=("Arial", 11, "bold"))
    label_edge_color.grid(row=4, column=0,  padx=2,  pady=2)
    entry_edge_color = ttk.Entry(root, width=20)
    entry_edge_color.grid(row=4, column=1,  padx=2,  pady=2)
    
    label_edge_width = ttk.Label(root, text="Edge Width:",foreground="blue", font=("Arial", 11, "bold"))
    label_edge_width.grid(row=4, column=2,  padx=2,  pady=2)
    entry_edge_width = ttk.Entry(root, width=10)
    entry_edge_width.grid(row=4, column=3,  padx=2,  pady=2)
    
    button_update_graph = ttk.Button(root, text="Update Graph", command=update_graph)
    button_update_graph.grid(row=4, column=4,  padx=2,  pady=2)
    
    label_centrality_threshold = ttk.Label(root, text="Classes Histogram:",foreground="blue", font=("Arial", 11, "bold"))
    label_centrality_threshold.grid(row=5, column=0,  padx=2,  pady=2)
    display_histogram_var = tk.BooleanVar()
    button_show_histogram = ttk.Checkbutton(root, text="Show Histogram", variable=display_histogram_var, command=execute_show_histogram)
    button_show_histogram.grid(row=5, column=1, columnspan=1,  padx=2,  pady=2)
    
    button_create_colored_graph = ttk.Button(root, text="Create Colored Graph", command=execute_create_colored_graph)
    button_create_colored_graph.grid(row=5, column=2,   padx=2, pady=2 ,columnspan=4)
    
    label_layout = ttk.Label(root, text="Change Layout:",foreground="blue", font=("Arial", 11, "bold"))
    label_layout.grid(row=6, column=0,  padx=2,  pady=2)   
    options = ['','spring', 'random', 'shell',  'circular', 'spectral', 'kamada_kawai', 'fruchterman_reingold', 'spiral', 'radial']
    selected_option = tk.StringVar(root)
    selected_option.set(options[0])  # Set the default option
    dropdown_layout = ttk.OptionMenu(root, selected_option, *options, command=change_layout)
    dropdown_layout.grid(row=6, column=1,  padx=2,  pady=2)
       
    
    label_centrality = ttk.Label(root, text="Choose Centrality:",foreground="blue", font=("Arial", 11, "bold"))
    label_centrality.grid(row=9, column=0,  padx=2,  pady=2)   
    filter_centrality_options = ['degree', 'betweenness', 'closeness']
    dropdown_centrality = tk.StringVar(root)
    dropdown_centrality.set(filter_centrality_options[0])  # Set default value     
    dropdown_centrality = tk.OptionMenu(root, dropdown_centrality, *filter_centrality_options)
    dropdown_centrality.grid(row=9, column=1,  padx=2,  pady=2)
    
    label_centrality_threshold = ttk.Label(root, text="Centrality Threshold:",foreground="blue", font=("Arial", 11, "bold"))
    label_centrality_threshold.grid(row=9, column=2,  padx=2,  pady=2)
    entry_centrality_threshold = ttk.Entry(root, width=10)
    entry_centrality_threshold.grid(row=9, column=3,  padx=2,  pady=2)
    
    filter_button = ttk.Button(root, text="Apply Centrality", command=filter_by_centrality)
    filter_button.grid(row=9, column=4,  padx=2,  pady=2)
   
    
    label_graph_statistics = ttk.Label(root, text="Graph Statistics:",foreground="blue", font=("Arial", 11, "bold"))
    label_graph_statistics.grid(row=8, column=0,   padx=2, pady=2 ,columnspan=1)
    display_statistics_var = tk.BooleanVar()
    compute_button = ttk.Checkbutton(root, text="Compute Graph Statistics", variable=display_statistics_var, command=create_graph_statistics)
    compute_button.grid(row=8, column=1,   padx=2, pady=2 ,columnspan=1)
    
   #new 
    label_graph_martrix = ttk.Label(root, text="Graph Metrics:",foreground="blue", font=("Arial", 11, "bold"))
    label_graph_martrix .grid(row=8, column=2,   padx=2, pady=2 ,columnspan=4)
    display_metrics_var = tk.BooleanVar()
    checkbox_display_metrics = ttk.Checkbutton(root, text="Display Graph Metrics", variable=display_metrics_var, command=toggle_graph_metrics)
    checkbox_display_metrics.grid(row=8, column=4,padx=10, pady=2 ,columnspan=4)
    
    
    label_membership = ttk.Label(root, text="Membership:",foreground="blue", font=("Arial", 11, "bold"))
    label_membership.grid(row=9, column=8,  padx=2,  pady=2)  
    membership_options = ['','Teachers', '1A', '2A',  '3A', '4A', '5A', '1B', '2B', '3B','4B','5B']    
    selected_membership = tk.StringVar(root)
    selected_membership.set(membership_options[0])  # Set the default option
    dropdown_membership = ttk.OptionMenu(root, selected_membership, *membership_options, command=filter_by_membership)
    dropdown_membership.grid(row=9, column=9,  padx=2,  pady=2)
    
    
    compute_button = ttk.Button(root, text="Compare Community Algorithm Results", command=compare_community_algorithms)
    compute_button.grid(row=10, column=1, columnspan=2,  padx=2,  pady=2)
    

    label_style = {"font": ("Arial", 11,"bold"), "fg": "blue"}
    checkbox_style = {"font": ("Arial", 9), "fg": "black"}
    # Clustering Using Louvain Algorithm
    
    checkbox_frame = Frame(root)
    checkbox_frame.grid(row=12, column=0,columnspan=2)
    
    Louvain_Clustering = Label(checkbox_frame, text="Clustering Using Louvain Algorithm by Clusterid: ", **label_style)
    Louvain_Clustering.grid(row=0, column=0, padx=2,  pady=2)
    
    Checkbutton(checkbox_frame, text="Undirected Graph", command=execute_louvain_undirected, **checkbox_style).grid(row=1, column=0, sticky="w")
    Checkbutton(checkbox_frame, text="Directed Graph", command=execute_louvain_directed, **checkbox_style).grid(row=2, column=0, sticky="w")
    
    # Basic Visualization
    checkbox_frame5 = Frame(root)
    checkbox_frame5.grid(row=12, column=3,columnspan=4)
    
    Basic_Visualization = Label(checkbox_frame5, text="Basic Visualization: ", **label_style)
    Basic_Visualization.grid(row=0, column=0,  padx=2,  pady=2)
    
    Checkbutton(checkbox_frame5, text="Undirected Graph", command=execute_basic_undirected, **checkbox_style).grid(row=1, column=0, sticky="w")
    Checkbutton(checkbox_frame5, text="Directed Graph", command=execute_basic_directed, **checkbox_style).grid(row=2, column=0, sticky="w")

    
 
    # Clustering by Using Degree
    checkbox_frame2 = Frame(root)
    checkbox_frame2.grid(row=13, column=0,columnspan=1)
    
    patitioning_degree = Label(checkbox_frame2, text="Clustering by Using Degree: ", **label_style)
    patitioning_degree.grid(row=0, column=0, padx=2,  pady=2)
    
    Checkbutton(checkbox_frame2, text="Undirected Graph", command=execute_partioning_by_degree_undirected, **checkbox_style).grid(row=1, column=0, sticky="w")
    Checkbutton(checkbox_frame2, text="Directed Graph", command=execute_partioning_by_degree_directed, **checkbox_style).grid(row=2, column=0, sticky="w")
    # Link Analysis Using Betweenness
    
    checkbox_frame7 = Frame(root)
    checkbox_frame7.grid(row=13, column=3,columnspan=5)
    
    Link_betweenness = Label(checkbox_frame7, text="Link Analysis Using Betweenness: ", **label_style)
    Link_betweenness.grid(row=0, column=0,  padx=2,  pady=2)
    
    Checkbutton(checkbox_frame7, text="Undirected Graph", command=execute_link_betweenness_undirected, **checkbox_style).grid(row=1, column=0, sticky="w")
    Checkbutton(checkbox_frame7, text="Directed Graph", command=execute_link_betweenness_directed, **checkbox_style).grid(row=2, column=0, sticky="w")
    
   
    
    # Clustering Using Girvan Newman and Degree
    checkbox_frame3 = Frame(root)
    checkbox_frame3.grid(row=14, column=0,columnspan=2)
    
    Louvain_Clustering = Label(checkbox_frame3, text="Clustering Using Girvan Newman and Degree: ", **label_style)
    Louvain_Clustering.grid(row=0, column=0, padx=2,  pady=2)
    
    Checkbutton(checkbox_frame3, text="Undirected Graph", command=execute_patrtioning_by_newman_and_degree, **checkbox_style).grid(row=1, column=0, sticky="w")
    Checkbutton(checkbox_frame3, text="Directed Graph", command=execute_patrtioning_by_newman_and_degree_directed, **checkbox_style).grid(row=2, column=0, sticky="w")

    
    
    # Link Analysis Using Pagerank
    checkbox_frame6 = Frame(root)
    checkbox_frame6.grid(row=14, column=3,columnspan=4)
    
    Link_Pagerank = Label(checkbox_frame6, text="Link Analysis Using Pagerank: ", **label_style)
    Link_Pagerank.grid(row=0, column=0, padx=2,  pady=2)
    
    Checkbutton(checkbox_frame6, text="Undirected Graph", command=execute_link_pagerank_undirected, **checkbox_style).grid(row=1, column=0, sticky="w")
    Checkbutton(checkbox_frame6, text="Directed Graph", command=execute_link_pagerank_directed, **checkbox_style).grid(row=2, column=0, sticky="w")

    
    
    # Clustering Using Girvan Newman Algorithm
    checkbox_frame4 = Frame(root)
    checkbox_frame4.grid(row=15, column=0,columnspan=2)
    
    Girvan_Newman = Label(checkbox_frame4, text="Clustering Using Girvan Newman Algorithm: ", **label_style)
    Girvan_Newman.grid(row=0, column=0, padx=2,  pady=2)
    
    Checkbutton(checkbox_frame4, text="Undirected Graph", command=execute_newman_undirected, **checkbox_style).grid(row=1, column=0, sticky="w")
    
    # Checkbutton(checkbox_frame1, text="Directed Graph", command=execute_newman_directed, **checkbox_style).grid(row=2, column=0, sticky="w")

    
    # Link Analysis Using Closeness
    checkbox_frame8 = Frame(root)
    checkbox_frame8.grid(row=15, column=3,columnspan=4)
    
    Link_Closeness = Label(checkbox_frame8, text="Link Analysis Using Closeness: ", **label_style)
    Link_Closeness.grid(row=0, column=0, padx=2,  pady=2)
    
    Checkbutton(checkbox_frame8, text="Undirected Graph", command=execute_link_closeness_undirected, **checkbox_style).grid(row=1, column=0, sticky="w")
    Checkbutton(checkbox_frame8, text="Directed Graph", command=execute_link_closeness_directed, **checkbox_style).grid(row=2, column=0, sticky="w")

    
    # Buttons for evaluation
    louvain_evalution =ttk.Button(root, text="Louvain Evaluation", command=execute_louvain_evalution)
    louvain_evalution.grid(row=20, column=1,columnspan=3, pady=5)
    
    newman_evalution = ttk.Button(root, text="Newman Evaluation", command=execute_newman_evalution)
    newman_evalution.grid(row=19, column=1,columnspan=3, pady=5)

    root.mainloop()
# Run the GUI
create_gui()
#################