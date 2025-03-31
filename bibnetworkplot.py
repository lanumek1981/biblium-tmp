# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:19:26 2023

@author: Lan.Umek
"""

import plotbib
import miscbib
import networkx as nx

from bibnetwork import BiblioNetwork

class PlotBiblioNetwork(BiblioNetwork):
    
    def plot_keyword_co_net(self, color_var="Average year of publication",
                        size_var="Number of documents", layout=nx.spring_layout, adjust_labels=True, **kwds):
    
        f0 = self.res_folder +"\\networks\\"
        f1, f2 = f0 + "keyword co-occurrence network", f0 + "keyword co-occurrence overlay"
        
        plotbib.plot_network(self.KW_cooc_net.net, part=self.KW_cooc_net.partition,
                             vec_size=self.KW_cooc_net.vectors[size_var], f_name=f1, 
                             dpi=self.dpi, adjust_labels=adjust_labels, layout=layout, **kwds)
        im1 = miscbib.BibImage(f1, name="keyword co-occurrence network")
        if im1 is not None:
            self.images.append(im1)
    
        plotbib.plot_network(self.KW_cooc_net.net, part=None, vec_color=self.KW_cooc_net.vectors[color_var],
                            vec_size=self.KW_cooc_net.vectors[size_var], f_name=f2,
                            dpi=self.dpi, adjust_labels=adjust_labels, layout=layout, **kwds)
        im2 = miscbib.BibImage(f2, name="keyword co-occurrence overlay")
        if im2 is not None:
            self.images.append(im2)
        
    def plot_keyword_co_heatmap(self, **kwds):
    
        f0 = self.res_folder +"\\plots\\"
        f1, f2 = f0 + "keyword co-occurrence heatmap", f0 + "normalized keyword co-occurrence heatmap"
        
        plotbib.heatmap_df(self.KW_cooc_net.co_df, label_cbar="Number of co-occurences", f_name=f1, square=True, **kwds)
        plotbib.heatmap_df(self.KW_cooc_net.co_s_df, label_cbar="Normalized co-occurences", f_name=f1, square=True, **kwds)
        
        im1 = miscbib.BibImage(f1, name="keyword co-occurrence heatmap")
        if im1 is not None:
            self.images.append(im1)
        im2 = miscbib.BibImage(f2, name="normalized keyword co-occurrence heatmap")
        if im2 is not None:
            self.images.append(im2)
            
    def thematic_plot_co_keyword(self, k=None, rnk=True, s_var="Number of documents", c_var="Average year"):
        # za s_var je druga opcija, privzeta v bibliometrixu: size
        net = self.KW_cooc_net
        if k is not None:
            net.clusters_df["show label"] = net.clusters_df["label"].map(lambda x: "\n".join(x.split("\n")[:k]))
        else:
            net.clusters_df["show label"] = net.clusters_df["label"]
        if rnk:
            x, y = "rank group betweenness centrality", "rank density"
        else:
            x, y = "group betweenness centrality", "density"

        f_name = self.res_folder +"\\plots\\thematic map"
        plotbib.scatter_plot_df(net.clusters_df, x, y, 
                        s_var=s_var, c_var=c_var, l_var="show label",                        
                        x_scale="linear", y_scale="linear", show_mean_line=True, 
                        x_label="relevance degree", y_label="development degree",
                        f_name=f_name, cmap=self.cmap,
                        remove_ticks=True)
        im = miscbib.BibImage(f_name, name="thematic map")
        if im is not None:
            self.images.append(im)
        