# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 08:23:35 2023

@author: Lan
"""

import plotbib
from bibgroup import BiblioGroup
from bibplot import BiblioPlot
from bibnetworkplot import PlotBiblioNetwork

class PlotBiblioGroup(BiblioGroup):
    
    def set_local_analysis(self):
        for v in self.g_values:
            df = self.df[self.df[self.split_var] == v]
            ba =  PlotBiblioNetwork(db=self.db, df=df)
            self.groups[v] = ba
            
    def plot_overlapping(self, label_cbar=None, kind="heatmap", subset=None,
                         include_totals=True, 
                         dissimilarity_metric='jaccard', linkage_method='ward', num_clusters=None, **kwds): # dodaj opcijo za shranjevanje
        if self.group_types == "values":
            print("Not relevant")
            return None
        if label_cbar is None:
            label_cbar= self.norm_method
        if kind == "heatmap":
            if self.ba.save_results:
                f_name = self.ba.res_folder +"\\plots\\overlap by heatmap "
            plotbib.heatmap_df(self.df_s_over_mat, label_cbar=label_cbar, xlabel="group", ylabel="group", square=True, f_name=f_name + "standardized")
            plotbib.heatmap_df(self.df_over_mat, label_cbar="Number of documents", xlabel="group", ylabel="group", square=True, 
                               fmt_cbar="{:d}", f_name=f_name + "absolute") # tole preveri format
        elif kind == "venn":
            if self.ba.save_results:
                f_name = self.ba.res_folder +"\\plots\\overlap by venn"
            if self.n_groups > 6:
                print("Too many groups")
                return None
            plotbib.plot_venn(self.g_ind_df, subset=subset, include_totals=include_totals, cmap=list(self.group_colors.values()), f_name=f_name, **kwds)
        elif kind == "dengrogram":
            if self.ba.save_results:
                f_name = self.ba.res_folder +"\\plots\\similarity by dengrogram"
            plotbib.hierarchical_clustering_pipeline(self.g_ind_df.T.reset_index(), dissimilarity_metric=dissimilarity_metric, linkage_method=linkage_method, num_clusters=num_clusters, f_name=f_name, dpi=self.ba.dpi)
        elif kind in ["MDS", "network", "clustermap"]:
            print("Not yet implemented")
            
    
    def plot_production(self, vrs1=None, vrs2=None, kind=None, title=None, colors=None, 
                        x_label=None, y_label=None, y_label_2=None,
                        before_str="before ", group_colors=None):
        
        if vrs1 is None:
            vrs1 = ["Number of documents (%s)" % v for v in self.g_values]
        if vrs2 is None:
            vrs2 = ["Cumulative number of citations (%s)" % v for v in self.g_values]     
        
        if self.ba.save_results:
            f_name = self.ba.res_folder +"\\plots\\time production over groups"
        else:
            f_name = None
        
        group_colors = list(self.group_colors.values()) if group_colors is None else group_colors
        
        if kind is None:
            if self.group_types == "values":
                kind = "bar"
            else:
                kind = "line"
                vrs2 = None
        
        plotbib.plot_production_groups(self.productions_df, vrs1, vrs2=vrs2, kind=kind,
                                        labels=self.g_values, colors=group_colors, f_name=f_name, 
                                        dpi=self.ba.dpi, title=title, 
                                        x_label=x_label, y_label=y_label, y_label_2=y_label_2,
                                        before_str=before_str)
        
    
    def plot_assoc_keywords(self, freq_var="a", color_var="yule_Q", f_name="wordcloud"):
        
        d = dict(zip(self.ba.keywords_df["Keyword"], self.ba.keywords_df["Number of documents"]))

        f_name0 = self.res_folder + "\\plots\\" + f_name

        for g_val in self.keywords_assoc_df["group"].unique():
            df_perf = self.keywords_assoc_df[self.keywords_assoc_df["group"]==g_val]
            df_perf["Number of documents"] = df_perf["x"].map(d)
            f_name = f_name0 + g_val
            if color_var == "yule_Q":
                label_color_var = "Yule Q for " + g_val
            plotbib.plot_word_cloud_from_freqs(df_perf, name_var="x", 
                                               freq_var=freq_var, color_var=color_var,
                                               label_color_var=label_color_var,
                                               round_scale=False, f_name=f_name)
            
    def tree_plot_assoc_keywords(self, freq_var="a", color_var="yule_Q", f_name="treemap", show=None, font_size=10):
        
        f_name0 = self.res_folder + "\\plots\\" + f_name
        
        for g_val in self.keywords_assoc_df["group"].unique():
            df_perf = self.keywords_assoc_df[self.keywords_assoc_df["group"]==g_val]
            f_name = f_name0 + g_val
            if color_var == "yule_Q":
                label_color_var = "Yule Q for " + g_val
                
            plotbib.plot_treemap(df_perf, size_var=freq_var, color_var=color_var,
                               labels_var="x", show=show, cmap_name=self.ba.cmap, 
                               ndig=self.ba.sig_digits, font_size=font_size, label_color_var=label_color_var,
                               f_name=f_name, dpi=self.ba.dpi)
            
      
    def plot_top_sources_barh(self, by="Number of documents", k=5, group_colors=None, title=None, 
                              label_var="Abbreviated Source Title", ncar=50, f_name="top sources barh", **kwds):   
        
        f_name0 = self.res_folder + "\\plots\\" + f_name
        
        group_colors = self.group_colors if group_colors is None else group_colors
        
        plotbib.create_grouped_top_k_barh_chart(self.sources_df, variable_column=by, group_column="group", 
                                                label_var=label_var,
                                                k=k, group_colors=group_colors, x_label=by, 
                                                y_label="Sources", chart_title=title, ncar=ncar,
                                                f_name=f_name0, dpi=self.ba.dpi, **kwds)
        
    def plot_top_ca_countries_barh(self, by="Number of documents", k=5, group_colors=None, title=None, 
                              label_var="CA Country", ncar=50, f_name="top ca countries barh", **kwds):   
        
        f_name0 = self.res_folder + "\\plots\\" + f_name
         
        group_colors = self.group_colors if group_colors is None else group_colors
        
        plotbib.create_grouped_top_k_barh_chart(self.ca_countries_df, variable_column=by, group_column="group", 
                                                label_var=label_var,
                                                k=k, group_colors=group_colors, x_label=by, 
                                                y_label="CA countries", chart_title=title, ncar=ncar,
                                                f_name=f_name0, dpi=self.ba.dpi, **kwds)
       
    def plot_top_authors_barh(self, by="Number of documents", k=5, group_colors=None, title=None, 
                              label_var="Author", ncar=50, f_name="top authors barh", **kwds):   
        
        f_name0 = self.res_folder + "\\plots\\" + f_name
            
        group_colors = self.group_colors if group_colors is None else group_colors
        
        plotbib.create_grouped_top_k_barh_chart(self.authors_df, variable_column=by, group_column="group", 
                                                label_var=label_var,
                                                k=k, group_colors=group_colors, x_label=by, 
                                                y_label="Authors", chart_title=title, ncar=ncar,
                                                f_name=f_name0, dpi=self.ba.dpi, **kwds)  
        
    def plot_top_ca_countries_stacked_barh(self, by="Number of documents", k=5, group_colors=None, title=None, 
                              label_var="CA Country", f_name="top ca countries barh stacked", show_percentages=False, perc_color="black", **kwds):   
        
        f_name0 = self.res_folder + "\\plots\\" + f_name
        
        group_colors = self.group_colors if group_colors is None else group_colors
        
        self.ca_countries_perc_df = plotbib.plot_stacked_barh(self.ca_countries_df, top=k, x=by, label=label_var, group="group",
                             show_percentages=show_percentages, perc_color=perc_color, group_colors=group_colors, title=title,
                             f_name=f_name0, dpi=self.ba.dpi)
        if self.save_results:
            self.ca_countries_perc_df.to_excel(self.res_folder + "\\tables\\percentage groups by ca country.xlsx")
        
        
    def scatter_plot_sources_g_prop(self, top=10, x="A", y="B", l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    f_name="sources scatter plot proportions", **kwds):
        
        df = self.sources_trans_df.head(top)
        x0, y0 = x, y
        x = [c for c in df.columns if "cond_g " in c if x in c][0]
        y = [c for c in df.columns if "cond_g " in c if y in c][0]
        
        f_name = self.res_folder + "\\plots\\" + f_name + f" {x0} {y0}"
        
        df["proportion in group " + x0], df["proportion in group " + y0] = df[x], df[y]
        
        plotbib.scatter_plot_df(df, "proportion in group " + x0, "proportion in group " + y0,
                                l_var=l_var, s_var=s_var, c_var=c_var, cmap=self.ba.cmap,
                                show_id_line=True, x_scale="linear", y_scale="linear",
                                f_name=f_name, **kwds)
        
    def scatter_plot_sources_g_prop_all(self, top=10, l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    f_name="sources scatter plot proportions", **kwds):
        for g in self.groups:
            for h in self.groups:
                if g != h:
                    self.scatter_plot_sources_g_prop(top=top, x=g, y=h, l_var=l_var,
                                                     s_var=s_var, c_var=c_var,
                                                     f_name=f_name, **kwds)
                    
                    
    def scatter_plot_authors_g_prop(self, top=10, x="A", y="B", l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    f_name="authors scatter plot proportions", **kwds):
        
        df = self.authors_trans_df.head(top)
        x0, y0 = x, y
        x = [c for c in df.columns if "cond_g " in c if x in c][0]
        y = [c for c in df.columns if "cond_g " in c if y in c][0]
        
        f_name = self.res_folder + "\\plots\\" + f_name + f" {x0} {y0}"
        
        df["proportion in group " + x0], df["proportion in group " + y0] = df[x], df[y]
        
        plotbib.scatter_plot_df(df, "proportion in group " + x0, "proportion in group " + y0,
                                l_var=l_var, s_var=s_var, c_var=c_var, cmap=self.ba.cmap,
                                show_id_line=True, x_scale="linear", y_scale="linear",
                                f_name=f_name, **kwds)
        
    def scatter_plot_authors_g_prop_all(self, top=10, l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    f_name="authors scatter plot proportions", **kwds):
        for g in self.groups:
            for h in self.groups:
                if g != h:
                    self.scatter_plot_authors_g_prop(top=top, x=g, y=h, l_var=l_var,
                                                     s_var=s_var, c_var=c_var,
                                                     f_name=f_name, **kwds)
                    
                    
    def scatter_plot_ca_countries_g_prop(self, top=10, x="A", y="B", l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    f_name="ca countries scatter plot proportions", **kwds):
        
        df = self.ca_countries_trans_df.head(top)
        x0, y0 = x, y
        x = [c for c in df.columns if "cond_g " in c if x in c][0]
        y = [c for c in df.columns if "cond_g " in c if y in c][0]
        
        f_name = self.res_folder + "\\plots\\" + f_name + f" {x0} {y0}"
        
        df["proportion in group " + x0], df["proportion in group " + y0] = df[x], df[y]
        
        plotbib.scatter_plot_df(df, "proportion in group " + x0, "proportion in group " + y0,
                                l_var=l_var, s_var=s_var, c_var=c_var, cmap=self.ba.cmap,
                                show_id_line=True, x_scale="linear", y_scale="linear",
                                f_name=f_name, **kwds)
        
    def scatter_plot_ca_countries_g_prop_all(self, top=10, l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    f_name="ca countries scatter plot proportions", **kwds):
        for g in self.groups:
            for h in self.groups:
                if g != h:
                    self.scatter_plot_ca_countries_g_prop(top=top, x=g, y=h, l_var=l_var,
                                                     s_var=s_var, c_var=c_var,
                                                     f_name=f_name, **kwds)

    def scatter_plot_keywords_g_prop(self, top=10, x="A", y="B", l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    show_k_lines=[],
                                    f_name="keywords scatter plot proportions", **kwds):
        
        df = self.keywords_trans_df.head(top)
        x0, y0 = x, y
        x = [c for c in df.columns if "cond_g " in c if x in c][0]
        y = [c for c in df.columns if "cond_g " in c if y in c][0]
        
        f_name = self.res_folder + "\\plots\\" + f_name + f" {x0} {y0}"
        
        df["proportion in group " + x0], df["proportion in group " + y0] = df[x], df[y]
        
        plotbib.scatter_plot_df(df, "proportion in group " + x0, "proportion in group " + y0,
                                l_var=l_var, s_var=s_var, c_var=c_var, cmap=self.ba.cmap,
                                show_id_line=True, show_k_lines=show_k_lines, x_scale="linear", y_scale="linear",
                                f_name=f_name, **kwds)
        
    def scatter_plot_keywords_g_prop_all(self, top=10, l_var="x",
                                    s_var="Total number of citations", c_var="Average year of publication",
                                    f_name="keywords scatter plot proportions", show_k_lines=[], **kwds):
        for g in self.groups:
            for h in self.groups:
                if g != h:
                    self.scatter_plot_keywords_g_prop(top=top, x=g, y=h, l_var=l_var,
                                                     s_var=s_var, c_var=c_var,
                                                     f_name=f_name, show_k_lines=show_k_lines, **kwds)
                    
    def heatmap_assoc_keywords(self, min_freq=5, measure="yule_Q", name="x", order=None,
                               annot=True, fmt=".2g", label_cbar=None, 
                                             fmt_cbar="%.2g", xticklabels=True, yticklabels=True,
                                             xlabel=None, ylabel=None, f_xlabel=10, f_ylabel=10,
                                             f_colorbar=10, f_lab_colorbar=10,
                                             annot_size=10, ylabel_size=10, xlabel_size=10,
                                             title=None, f_name="associated keywords heatmap", dpi=1200, **kwds):
        if not hasattr(self, "keywords_trans_df"):
            print("Compute associations first")
            return None
        
        df = self.keywords_trans_df
        if min_freq is not None:
            df = df[df["Number of documents"] >= min_freq]
        
        f_name = self.res_folder + "\\plots\\" + f_name    
        
        plotbib.plot_assocations_df(df, measure=measure, name=name, order=order, cmap=self.ba.cmap,
                                    annot=annot, fmt=fmt, label_cbar=label_cbar, 
                                    fmt_cbar=fmt_cbar, xticklabels=xticklabels, yticklabels=yticklabels,
                                    xlabel=xlabel, ylabel=ylabel, f_xlabel=f_xlabel, f_ylabel=f_ylabel, 
                                    f_colorbar=f_colorbar, f_lab_colorbar=f_lab_colorbar,
                                    annot_size=annot_size, ylabel_size=ylabel_size, xlabel_size=xlabel_size,
                                    title=title, f_name=f_name, dpi=self.ba.dpi, **kwds)
        
    
    def heatmap_assoc_sources(self, min_freq=5, measure="yule_Q", name="x", order=None,
                               annot=True, fmt=".2g", label_cbar=None, 
                                             fmt_cbar="%.2g", xticklabels=True, yticklabels=True,
                                             xlabel=None, ylabel=None, f_xlabel=10, f_ylabel=10,
                                             f_colorbar=10, f_lab_colorbar=10,
                                             annot_size=10, ylabel_size=10, xlabel_size=10,
                                             title=None, f_name="associated sources heatmap", dpi=1200, **kwds):
        if not hasattr(self, "sources_trans_df"):
            print("Compute associations first")
            return None
        
        df = self.sources_trans_df
        if min_freq is not None:
            df = df[df["Number of documents"] >= min_freq]
        
        f_name = self.res_folder + "\\plots\\" + f_name    
        
        plotbib.plot_assocations_df(df, measure=measure, name=name, order=order, cmap=self.ba.cmap,
                                    annot=annot, fmt=fmt, label_cbar=label_cbar, 
                                    fmt_cbar=fmt_cbar, xticklabels=xticklabels, yticklabels=yticklabels,
                                    xlabel=xlabel, ylabel=ylabel, f_xlabel=f_xlabel, f_ylabel=f_ylabel, 
                                    f_colorbar=f_colorbar, f_lab_colorbar=f_lab_colorbar,
                                    annot_size=annot_size, ylabel_size=ylabel_size, xlabel_size=xlabel_size,
                                    title=title, f_name=f_name, dpi=self.ba.dpi, **kwds)  
        
    def heatmap_assoc_ca_countries(self, min_freq=5, measure="yule_Q", name="x", order=None,
                               annot=True, fmt=".2g", label_cbar=None, 
                                             fmt_cbar="%.2g", xticklabels=True, yticklabels=True,
                                             xlabel=None, ylabel=None, f_xlabel=10, f_ylabel=10,
                                             f_colorbar=10, f_lab_colorbar=10,
                                             annot_size=10, ylabel_size=10, xlabel_size=10,
                                             title=None, f_name="associated ca countries heatmap", dpi=1200, **kwds):
        if not hasattr(self, "ca_countries_trans_df"):
            print("Compute associations first")
            return None
        
        df = self.ca_countries_trans_df
        if min_freq is not None:
            df = df[df["Number of documents"] >= min_freq]
        
        f_name = self.res_folder + "\\plots\\" + f_name    
        
        plotbib.plot_assocations_df(df, measure=measure, name=name, order=order, cmap=self.ba.cmap,
                                    annot=annot, fmt=fmt, label_cbar=label_cbar, 
                                    fmt_cbar=fmt_cbar, xticklabels=xticklabels, yticklabels=yticklabels,
                                    xlabel=xlabel, ylabel=ylabel, f_xlabel=f_xlabel, f_ylabel=f_ylabel, 
                                    f_colorbar=f_colorbar, f_lab_colorbar=f_lab_colorbar,
                                    annot_size=annot_size, ylabel_size=ylabel_size, xlabel_size=xlabel_size,
                                    title=title, f_name=f_name, dpi=self.ba.dpi, **kwds)
    
    def heatmap_assoc_authors(self, min_freq=5, measure="yule_Q", name="x", order=None,
                               annot=True, fmt=".2g", label_cbar=None, 
                                             fmt_cbar="%.2g", xticklabels=True, yticklabels=True,
                                             xlabel=None, ylabel=None, f_xlabel=10, f_ylabel=10,
                                             f_colorbar=10, f_lab_colorbar=10,
                                             annot_size=10, ylabel_size=10, xlabel_size=10,
                                             title=None, f_name="associated authors heatmap", dpi=1200, **kwds):
        if not hasattr(self, "authors_trans_df"):
            print("Compute associations first")
            return None
        
        df = self.authors_trans_df
        if min_freq is not None:
            df = df[df["Number of documents"] >= min_freq]
        
        f_name = self.res_folder + "\\plots\\" + f_name    
        
        plotbib.plot_assocations_df(df, measure=measure, name=name, order=order, cmap=self.ba.cmap,
                                    annot=annot, fmt=fmt, label_cbar=label_cbar, 
                                    fmt_cbar=fmt_cbar, xticklabels=xticklabels, yticklabels=yticklabels,
                                    xlabel=xlabel, ylabel=ylabel, f_xlabel=f_xlabel, f_ylabel=f_ylabel, 
                                    f_colorbar=f_colorbar, f_lab_colorbar=f_lab_colorbar,
                                    annot_size=annot_size, ylabel_size=ylabel_size, xlabel_size=xlabel_size,
                                    title=title, f_name=f_name, dpi=self.ba.dpi, **kwds)  
        
        
                    
    def basic_analysis(self, methods=None):
        
        if methods is None:
            methods = [self.get_main_info, self.get_production, self.plot_production, self.get_top_cited_docs,
                      self.count_sources, self.count_ca_countries, self.count_authors,
                      self.count_keywords, self.get_sources_stats, self.get_ca_countries_stats,
                      self.get_authors_stats, self.associate_sources,  self.associate_ca_countries,
                      self.associate_keywords, self.associate_authors, self.to_excel]
        
        for m in methods:
            try:
                m()
            except:
                print("Problem with running ", m)

            
    #def plot_keyword_co_net(self, **kwds):
    #    for g_val in self.g_values:
    #        self.groups[g_val].plot_keyword_co_net(**kwds)