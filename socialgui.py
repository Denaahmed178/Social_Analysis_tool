
from tkinter import *
from tkinter.ttk import Separator

import pandas as pd
import socialTask

nodes = pd.read_csv('Nodes.csv')
edges = pd.read_csv('Edges.csv')


def execute_louvain_evalution():
    socialTask.evalution_louvain_G(nodes=nodes,edges=edges)

def execute_newman_evalution():
    socialTask.evalution_newman_G(nodes=nodes,edges=edges)
def execute_louvain_directed():
    socialTask.partition_louvain_directed_clusterid(nodes=nodes, edges=edges)

def execute_louvain_undirected():
    socialTask.partition_louvain_clusterid(nodes=nodes, edges=edges)

def execute_partioning_by_degree_undirected():
    socialTask.partition_graph_by_degree(nodes=nodes,edges=edges,bins=6)

def execute_partioning_by_degree_directed():
    socialTask.partition_graph_by_degree_directed(nodes=nodes,edges=edges,bins=6)

def execute_patrtioning_by_newman_and_degree():
    socialTask.newman_partitioning_by_degree_bins(nodes=nodes,edges=edges,num_bins=4)

def execute_patrtioning_by_newman_and_degree_directed():
    socialTask.newman_partitioning_by_degree_bins_directed(nodes=nodes,edges=edges,num_bins=4)

def execute_newman_undirected():
    socialTask.girvan_newman_algoritm_undirected(nodes=nodes, edges=edges)

def execute_basic_undirected():
    socialTask.basic_undirected_G(nodes=nodes, edges=edges)

def execute_basic_directed():
    socialTask.basic_directed_G(nodes=nodes, edges=edges)

def execute_link_pagerank_undirected():
    socialTask.link_pagerank_undirected(nodes=nodes, edges=edges)

def execute_link_pagerank_directed():
    socialTask.link_pagerank_directed(nodes=nodes, edges=edges)

def execute_link_betweenness_undirected():
    socialTask.link_betweenness_undirected(nodes=nodes, edges=edges)

def execute_link_betweenness_directed():
    socialTask.link_betweenness_directed(nodes=nodes, edges=edges)

def execute_link_closeness_undirected():
    socialTask.link_closeness_undirected(nodes=nodes, edges=edges)

def execute_link_closeness_directed():
    socialTask.link_closeness_directed(nodes=nodes, edges=edges)


root = Tk()


# Define a common style for labels and checkboxes
label_style = {"font": ("Arial", 20), "fg": "#800080", "padx": 20, "pady": 5}
checkbox_style = {"font": ("Arial", 12), "fg": "black", "padx": 70, "pady": 5}


# Create a new window
checkbox_frame = Frame(root)
checkbox_frame.pack(fill="x")

# Create and display checkboxes
Louvain_Clustering = Label(checkbox_frame, text="Clustering Using Louvain Algorithm by Clusterid: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame, text="Undirected Graph", command=execute_louvain_undirected, **checkbox_style).pack(anchor="w")
Checkbutton(checkbox_frame, text="Directed Graph", command=execute_louvain_directed, **checkbox_style).pack(anchor="w")

sep1 = Separator(root, orient='horizontal')
sep1.pack(fill='x', pady=10)
####################################################################################################################################

# Create a new window
checkbox_frame6 = Frame(root)
checkbox_frame6.pack(fill="x")

# Create and display checkboxes
patitioning_degree = Label(checkbox_frame6, text="Clustering by Using Degree: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame6, text="Undirected Graph", command=execute_partioning_by_degree_undirected, **checkbox_style).pack(anchor="w")
Checkbutton(checkbox_frame6, text="Directed Graph", command=execute_partioning_by_degree_directed, **checkbox_style).pack(anchor="w")

sep6 = Separator(root, orient='horizontal')
sep6.pack(fill='x', pady=10)

# Create a new window
checkbox_frame7 = Frame(root)
checkbox_frame7.pack(fill="x")

###################################################################################################################################
# Create and display checkboxes
Louvain_Clustering = Label(checkbox_frame7, text="Clustering Using Girvan Newman and Degree: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame7, text="Undirected Graph", command=execute_patrtioning_by_newman_and_degree, **checkbox_style).pack(anchor="w")
Checkbutton(checkbox_frame7, text="Directed Graph", command=execute_patrtioning_by_newman_and_degree_directed, **checkbox_style).pack(anchor="w")

sep7 = Separator(root, orient='horizontal')
sep7.pack(fill='x', pady=10)

checkbox_frame1 = Frame(root)
checkbox_frame1.pack(fill="x")

# Create and display checkboxes
Girvan_Newman = Label(checkbox_frame1, text="Clustering Using Girvan Newman Algorithm: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame1, text="Undirected Graph", command=execute_newman_undirected, **checkbox_style).pack(anchor="w")
# Checkbutton(checkbox_frame1, text="Directed Graph", command=execute_newman_directed, **checkbox_style).pack(anchor="w")

sep2 = Separator(root, orient='horizontal')
sep2.pack(fill='x', pady=10)
####################################################################################################################################
checkbox_frame2 = Frame(root)
checkbox_frame2.pack(fill="x")

# Create and display checkboxes
Basic_Visualization = Label(checkbox_frame2, text="Basic Visualization: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame2, text="Undirected Graph", command=execute_basic_undirected, **checkbox_style).pack(anchor="w")
Checkbutton(checkbox_frame2, text="Directed Graph", command=execute_basic_directed, **checkbox_style).pack(anchor="w")

sep3 = Separator(root, orient='horizontal')
sep3.pack(fill='x', pady=10)
####################################################################################################################################
checkbox_frame3 = Frame(root)
checkbox_frame3.pack(fill="x")

# Create and display checkboxes
Link_Pagerank = Label(checkbox_frame3, text="Link Analysis Using Pagerank: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame3, text="Undirected Graph", command=execute_link_pagerank_undirected, **checkbox_style).pack(anchor="w")
Checkbutton(checkbox_frame3, text="Directed Graph", command=execute_link_pagerank_directed, **checkbox_style).pack(anchor="w")

sep4 = Separator(root, orient='horizontal')
sep4.pack(fill='x', pady=10)
####################################################################################################################################
checkbox_frame4 = Frame(root)
checkbox_frame4.pack(fill="x")

# Create and display checkboxes
Link_betweenness = Label(checkbox_frame4, text="Link Analysis Using Betweenness: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame4, text="Undirected Graph", command=execute_link_betweenness_undirected, **checkbox_style).pack(anchor="w")
Checkbutton(checkbox_frame4, text="Directed Graph", command=execute_link_betweenness_directed, **checkbox_style).pack(anchor="w")

sep5 = Separator(root, orient='horizontal')
sep5.pack(fill='x', pady=10)
####################################################################################################################################
checkbox_frame5 = Frame(root)
checkbox_frame5.pack(fill="x")

# Create and display checkboxes
Link_Closeness = Label(checkbox_frame5, text="Link Analysis Using Closeness: ", **label_style).pack(anchor="w")
Checkbutton(checkbox_frame5, text="Undirected Graph", command=execute_link_closeness_undirected, **checkbox_style).pack(anchor="w")
Checkbutton(checkbox_frame5, text="Directed Graph", command=execute_link_closeness_directed, **checkbox_style).pack(anchor="w")

sep8 = Separator(root, orient='horizontal')
sep8.pack(fill='x', pady=10)


louvain_evalution = Button(root, text="Louvain Evalution", command=execute_louvain_evalution, width="30", bg="black", fg="white", font=("Arial", 12))
louvain_evalution.pack()

newman_evalution = Button(root, text=" Newman Evalution", command=execute_newman_evalution, width="30", bg="black", fg="white", font=("Arial", 12))
newman_evalution.pack()
root.mainloop()
