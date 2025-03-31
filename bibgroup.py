# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:15:08 2023

@author: Lan
"""

import pandas as pd
import numpy as np
import os
import readbib, utilsbib, plotbib, reportbib, miscbib

from functools import reduce


from bibstats import BiblioStats, my_output


class BiblioGroup:
    
    def __init__(self, f_name=None, db="", df=None, bib_file=None, split_var=None, ind_df=None, value_order=None,
                 res_folder="results - groups", save_results=True, preprocess=0,
                 norm_method = "jaccard",
                  measures=["jaccard", "yule_Q"], # dodaj cond_x in cond_y
                  adjust_p="fdr_bh", alpha=0.1, verbose=3,
                  group_colors=None):
        
        if bib_file is not None:
            self.df, self.db = bib_file
        else:
            self.db = db.lower()
            self.df = miscbib.read_data(f_name=f_name, db=self.db, df=df)
        
        if self.db not in ["scopus", "wos"]:
            print("No database info provided")
            return None
        
       
        self.verbose = verbose
        self.save_results = save_results
        sub_folders = ["plots", "tables", "reports", "indicators", "networks"] 
        self.res_folder = os.getcwd() + "\\" + res_folder
        if self.save_results:
            folders = [self.res_folder + "\\" + s for s in sub_folders]
            utilsbib.make_folders(folders)

        self.ba = BiblioStats(f_name=f_name, db=db, df=df, bib_file=bib_file,
                              res_folder=res_folder, verbose=0, preprocess=preprocess) # preveri za ta folder 
        self.ba.get_mappings()
        #self.ldf, self.ldd = self.ba.ldf, self.ba.ldd
        self.groups = {}
        
        
        if split_var is not None:
            self.group_types = "values"
            self.split_var, self.df = split_var, self.df[self.df[split_var].notna()]
            self.g_ind_df = utilsbib.get_indicator_df(self.df, split_var, name="group")
            
            if value_order is None:
                self.g_values = self.df[self.split_var].unique() 
                self.g_values.sort()
            else:
                self.g_values = np.array(value_order)
            for v in self.g_values:
                df = self.df[self.df[self.split_var] == v]
                ba = BiblioStats(db=db, df=df) 
                self.groups[v] = ba
        elif ind_df is not None:
            self.group_types = "indicators"
            self.g_ind_df, self.g_values = ind_df, ind_df.columns
            for c in ind_df.columns:
                df = self.df[ind_df[c]==1]
                ba = BiblioStats(f_name=f_name, db=db, df=df, bib_file=bib_file) 
                self.groups[c] = ba
                                
            self.over_mat = np.dot(ind_df.T, ind_df)
            mat_diag = np.diag(self.over_mat)
            self.norm_method = norm_method
            self.s_over_mat = utilsbib.normalize_sq_matrix(self.over_mat, method=self.norm_method)
            self.df_over_mat, self.df_s_over_mat = pd.DataFrame(self.over_mat), pd.DataFrame(self.s_over_mat)
            self.df_over_mat.index, self.df_over_mat.columns = ind_df.columns, ind_df.columns
            self.df_s_over_mat.index, self.df_s_over_mat.columns = ind_df.columns, ind_df.columns
        else:
            print("Group must be defined")
            return None

        
        self.Y = self.g_ind_df.reset_index(drop=True)
        self.measures = measures
        self.n_groups = len(self.groups)
        self.adjust_p, self.alpha = adjust_p, alpha

        
        if group_colors is None:
            group_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] + ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] + ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"] 
            #def hex_to_rgba(hex_color):
            #    hex_color = hex_color.lstrip('#')
            #    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (1.0,)

            #rgba_colors = [hex_to_rgba(color) for color in gc]
            #self.group_colors = {group: rgba_colors[i] for i, group in enumerate(self.groups.keys())}
            
            self.group_colors = {group: group_colors[i] for i, group in enumerate(self.groups.keys())}
        else:
            self.group_colors = group_colors
        
    
    def save_to_file(self, file_name="biblio-group.pkl", exclude_dataset=True):
        with open(self.res_folder + "\\" + file_name, "wb") as file:
            import pickle
            if exclude_dataset:
                df0 = self.df
                self.df = None
                for g in self.groups:
                    self.groups[g].df=None
                    
            pickle.dump(self, file)
            self.df = df0
        print(f"Analysis saved to {file_name}")    

            
    def get_main_info(self, f_name="main info"):
        self.main_info = []
        for v in self.g_values:
            try:
                self.groups[v].get_main_info(f_name=None)
                vals = [v] + self.groups[v].main_info_df["value"].values.tolist() # tukaj bo morda treba spremeniti ime stolpca iz value v kaj drugega
                self.main_info.append(vals)
            except:
                print("Problem")
        if self.group_types == "values":
            cn = self.split_var
        else:
            cn = "group"
        cols = [cn] + self.groups[v].main_info_df["key"].values.tolist() # enako tukaj key x kaj drugega
        self.main_info_df = pd.DataFrame(self.main_info, columns=cols)
        self.main_info_df = self.main_info_df.T.reset_index()
        my_output(self, "main_info_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    """    
    def get_production(self, fill_empty=True, rng=None, cut_year=None, exclude_last="smart", f_name="scientific group production"):
        if self.group_types != "values":
            print("Works for groups defined by a single splitting variable")
            return None
        for v in self.g_values:
            self.groups[v].get_production(fill_empty=fill_empty, rng=rng, cut_year=cut_year, exclude_last=exclude_last, f_name=f_name)
            self.groups[v].production_df = self.groups[v].production_df.rename(
                columns={"Number of documents": "Number of documents (%s)" %v,
                         "Cited by": "Cited by (%s)" %v,
                         "Cummulative number of documents": "Cummulative number of documents (%s)" %v,
                         "Cummulative number of citations": "Cummulative number of citations (%s)" %v,
                         "Average citations per document": "Average citations per document (%s)" %v})
            
            
        production_frames = [self.groups[v].production_df for v in self.g_values]
        self.productions_df = reduce(lambda  left, right: pd.merge(
            left, right, on=["Year"], how="outer"), production_frames)
        
        self.productions_df["Number of documents together"] = self.productions_df[[c for c in self.productions_df.columns if "Number of documents (" in c]].sum(axis=1)
        for v in self.g_values:
            self.productions_df["Proportion of documents in group (%s)" %v] = self.productions_df["Number of documents (%s)" %v] / self.productions_df["Number of documents together"]
    """    

    def get_production(self, fill_empty=True, rng=None, cut_year=None, exclude_last="smart", f_name="scientific group production"):

        for v in self.g_values:
            self.groups[v].get_production(fill_empty=fill_empty, rng=rng, cut_year=cut_year, exclude_last=exclude_last, f_name=f_name)
            self.groups[v].production_df = self.groups[v].production_df.rename(
                columns={"Number of documents": "Number of documents (%s)" %v,
                         "Cited by": "Cited by (%s)" %v,
                         "Cumulative number of documents": "Cumulative number of documents (%s)" %v,
                         "Cumulative number of citations": "Cumulative number of citations (%s)" %v,
                         "Average citations per document": "Average citations per document (%s)" %v})
            
            
        production_frames = [self.groups[v].production_df for v in self.g_values]
        self.productions_df = reduce(lambda  left, right: pd.merge(
            left, right, on=["Year"], how="outer"), production_frames)
        if self.group_types != "values":
            self.productions_df["Number of documents together"] = self.productions_df[[c for c in self.productions_df.columns if "Number of documents (" in c]].sum(axis=1)
            for v in self.g_values:
                self.productions_df["Proportion of documents in group (%s)" %v] = self.productions_df["Number of documents (%s)" %v] / self.productions_df["Number of documents together"]
        
        
    def get_top_cited_docs(self, top=10, where="global", limit_to=None, add_vars=[], rm_vars=[], f_name="top cited documents"):
        gl_docs, loc_docs = [], []
        for v in self.g_values:
            self.groups[v].get_top_cited_docs(top=top, where=where, limit_to=limit_to, 
                              add_vars=add_vars, rm_vars=rm_vars)
            if hasattr(self.groups[v], "top_gl_cit_docs_df"):
                d = self.groups[v].top_gl_cit_docs_df
                d["group"] = v
                gl_docs.append(d)
            # LOCAL: TO-DO
        if len(gl_docs):
            self.top_gl_cit_docs_df = pd.concat(gl_docs)
        if len(loc_docs):
            self.top_loc_cit_docs_df = pd.concat(loc_docs)
        my_output(self, "top_gl_cit_docs_df", f_name="\\tables\\" + f_name + ".xlsx")
 
    def count_sources(self, f_name="sources counts", trans=["Number of documents", "Proportion of documents", "Relative rank"]):
        for v in self.g_values:
            self.groups[v].count_sources()
            self.groups[v].sources_df["group"] = v
        self.sources_df = pd.concat([self.groups[v].sources_df for v in self.g_values], axis=0)
        my_output(self, "sources_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_sources_stats()")
        if len(trans):
             self.sources_trans_df = miscbib.transform_group_df(self.sources_df, name_var="Source title", group_var="group", vs=trans)
             my_output(self, "sources_trans_df", f_name="\\tables\\" + f_name + " trans.xlsx")
             
    def count_areas(self, f_name="sources counts", trans=['Number of documents', "Fractional count of documents", "Proportion of documents", "Fractional proportion of documents", "Relative rank"]):
        for v in self.g_values:
            self.groups[v].count_areas()
            self.groups[v].areas_df["group"] = v
            self.groups[v].fields_df["group"] = v
            self.groups[v].sciences_df["group"] = v
        self.areas_df = pd.concat([self.groups[v].areas_df for v in self.g_values], axis=0)
        self.fields_df = pd.concat([self.groups[v].fields_df for v in self.g_values], axis=0)
        self.sciences_df = pd.concat([self.groups[v].sciences_df for v in self.g_values], axis=0)
        #my_output(self, "sources_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_sources_stats()")
        if len(trans):
             self.areas_trans_df = miscbib.transform_group_df(self.areas_df, name_var="Area", group_var="group", vs=trans)
             self.fields_trans_df = miscbib.transform_group_df(self.fields_df, name_var="Field", group_var="group", vs=trans)
             self.sciences_trans_df = miscbib.transform_group_df(self.sciences_df, name_var="Science", group_var="group", vs=trans)
             #my_output(self, "sources_trans_df", f_name="\\tables\\" + f_name + " trans.xlsx")

        
    def count_keywords(self, f_name="keywords counts", trans=["Number of documents", "Proportion of documents", "Relative rank"]):
        for v in self.g_values:
            self.groups[v].count_keywords()
            self.groups[v].keywords_df["group"] = v
        self.keywords_df = pd.concat([self.groups[v].keywords_df for v in self.g_values], axis=0)
        my_output(self, "keywords_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_keywords_stats()")
        if len(trans):
             self.keywords_trans_df = miscbib.transform_group_df(self.keywords_df, name_var="Keyword", group_var="group", vs=trans)
             my_output(self, "keywords_trans_df", f_name="\\tables\\" + f_name + " trans.xlsx")
             
    def count_ca_countries(self, f_name="ca countries counts", trans=["Number of documents", "Proportion of documents", "Relative rank"]):
        for v in self.g_values:
            self.groups[v].count_ca_countries()
            self.groups[v].ca_countries_df["group"] = v
        self.ca_countries_df = pd.concat([self.groups[v].ca_countries_df for v in self.g_values], axis=0)
        my_output(self, "ca_countries_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_ca_countries_stats()")
        if len(trans):
             self.ca_countries_trans_df = miscbib.transform_group_df(self.ca_countries_df, name_var="CA Country", group_var="group", vs=trans)
             my_output(self, "ca_countries_trans_df", f_name="\\tables\\" + f_name + " trans.xlsx")

    def count_authors(self, f_name="authors counts", trans=['Number of documents', "Fractional count of documents", "Proportion of documents", "Fractional proportion of documents", "Relative rank"]):
        for v in self.g_values:
            self.groups[v].count_authors()
            self.groups[v].authors_df["group"] = v
        self.authors_df = pd.concat([self.groups[v].authors_df for v in self.g_values], axis=0)
        my_output(self, "authors_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_authors_stats()")
        if len(trans):
             self.authors_trans_df = miscbib.transform_group_df(self.authors_df, name_var="Author(s) ID", group_var="group", vs=trans)
             my_output(self, "authors_trans_df", f_name="\\tables\\" + f_name + " trans.xlsx")


    def get_sources_stats(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top sources stats"):
        for v in self.g_values:
            self.groups[v].get_sources_stats(items=items, exclude=exclude, 
                          top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop,
                          level=level, compute_indicators=compute_indicators,
                          abbreviate=abbreviate, f_name=None)
            self.groups[v].sources_df["group"] = v
        self.sources_df = pd.concat([self.groups[v].sources_df for v in self.g_values])
        my_output(self, "sources_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def get_authors_stats(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top authors stats"):
        for v in self.g_values:
            self.groups[v].get_authors_stats(items=items, exclude=exclude, 
                          top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop,
                          level=level, compute_indicators=compute_indicators, f_name=None)
            self.groups[v].authors_df["group"] = v
        self.authors_df = pd.concat([self.groups[v].authors_df for v in self.g_values])
        my_output(self, "authors_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def get_ca_countries_stats(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top ca countries stats"):
        for v in self.g_values:
            self.groups[v].get_ca_countries_stats(items=items, exclude=exclude, 
                          top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop,
                          level=level, compute_indicators=compute_indicators, f_name=None)
            self.groups[v].ca_countries_df["group"] = v
        self.ca_countries_df = pd.concat([self.groups[v].ca_countries_df for v in self.g_values])
        my_output(self, "ca_countries_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def get_keywords_stats(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top keywords stats"):
        for v in self.g_values:
            self.groups[v].get_keywords_stats(items=items, exclude=exclude, 
                          top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop,
                          level=level, compute_indicators=False, f_name=None)
            self.groups[v].keywords_df["group"] = v
        if compute_indicators:
            self.ba.get_keywords_stats(
                items=items, exclude=exclude, top=top, top_by=top_by,
                           min_freq=min_freq, cov_prop=cov_prop, 
                          freq_col="Number of documents", 
                          level=level, compute_indicators=True)
            self.keywords_df_ind = self.ba.keywords_df_ind
        self.keywords_df = pd.concat([self.groups[v].keywords_df for v in self.g_values])
        my_output(self, "keywords_df", f_name="\\tables\\" + f_name + ".xlsx")
    
    
    def associate_sources(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True,
                          abbreviate=True, f_name="associated sources", sort_by="a",
                          transform=True, vs=["yule_Q", "cond_g"]):
        self.ba.get_mappings()
        self.ba.get_sources_stats(items=items, exclude=exclude, top=top,
                                  top_by=top_by, min_freq=min_freq, cov_prop=cov_prop,
                                  level=1, compute_indicators=True, abbreviate=True)
        self.sources_assoc_df = utilsbib.associate_dfs(
            self.g_ind_df, self.ba.sources_df_ind, measures=self.measures, sort_by=sort_by, 
            correction=self.adjust_p, alpha=self.alpha)
        
        if transform:
            dt = miscbib.transform_group_df(self.sources_assoc_df, name_var="x", group_var="group", vs=vs)
            df0 = self.ba.sources_df
            if abbreviate:
                df0["x"] = df0["Abbreviated Source Title"]
            else:
                df0["x"] = df0["Source title"]
            self.sources_trans_df = pd.merge(dt, df0, on="x", how="left")
            self.sources_trans_df= self.sources_trans_df.sort_values("Number of documents", ascending=False)
            my_output(self, "sources_trans_df", f_name="\\tables\\" + f_name + " transformed.xlsx")
        my_output(self, "sources_assoc_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def associate_authors(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True, 
                          f_name="associated authors", sort_by="a",
                          transform=True, vs=["yule_Q", "cond_g"]):
        self.ba.get_authors_stats(items=items, exclude=exclude, top=top,
                                  top_by=top_by, min_freq=min_freq, cov_prop=cov_prop,
                                  level=1, compute_indicators=True)
        self.authors_assoc_df = utilsbib.associate_dfs(
            self.g_ind_df, self.ba.authors_df_ind, measures=self.measures, sort_by=sort_by, 
            correction=self.adjust_p, alpha=self.alpha)
        if transform:
            dt = miscbib.transform_group_df(self.authors_assoc_df, name_var="x", group_var="group", vs=vs)
            df0 = self.ba.authors_df
            df0["x"] = df0["Author"]
            self.authors_trans_df = pd.merge(dt, df0, on="x", how="left")
            self.authors_trans_df= self.authors_trans_df.sort_values("Number of documents", ascending=False)
            my_output(self, "authors_trans_df", f_name="\\tables\\" + f_name + " transformed.xlsx")
        my_output(self, "authors_assoc_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def associate_ca_countries(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True, 
                          f_name="associated ca countries", sort_by="a",
                          transform=True, vs=["yule_Q", "cond_g"]):
        self.ba.get_ca_countries_stats(items=items, exclude=exclude, top=top,
                                  top_by=top_by, min_freq=min_freq, cov_prop=cov_prop,
                                  level=1, compute_indicators=True)
        self.ca_countries_assoc_df = utilsbib.associate_dfs(
            self.g_ind_df, self.ba.ca_countries_df_ind, measures=self.measures, sort_by=sort_by, 
            correction=self.adjust_p, alpha=self.alpha) 
        if transform:
            dt = miscbib.transform_group_df(self.ca_countries_assoc_df, name_var="x", group_var="group", vs=vs)
            df0 = self.ba.ca_countries_df
            df0["x"] = df0["CA Country"]
            self.ca_countries_trans_df = pd.merge(dt, df0, on="x", how="left")
            self.ca_countries_trans_df= self.ca_countries_trans_df.sort_values("Number of documents", ascending=False)
            my_output(self, "ca_countries_trans_df", f_name="\\tables\\" + f_name + " transformed.xlsx")
        my_output(self, "ca_countries_assoc_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def associate_keywords(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True, 
                          f_name="associated keywords", sort_by="a",
                          transform=True, vs=["yule_Q", "cond_g"]):
        self.ba.count_keywords()
        self.ba.get_keywords_stats(items=items, exclude=exclude, top=top,
                                  top_by=top_by, min_freq=min_freq, cov_prop=cov_prop, 
                                  level=1, compute_indicators=True)
        self.keywords_assoc_df = utilsbib.associate_dfs(
            self.g_ind_df, self.ba.keywords_df_ind, measures=self.measures, sort_by=sort_by, 
            correction=self.adjust_p, alpha=self.alpha)
        if transform:
            dt = miscbib.transform_group_df(self.keywords_assoc_df, name_var="x", group_var="group", vs=vs)
            df0 = self.ba.keywords_df
            df0["x"] = df0["Keyword"]
            self.keywords_trans_df = pd.merge(dt, df0, on="x", how="left")
            self.keywords_trans_df = self.keywords_trans_df.dropna()
            self.keywords_trans_df= self.keywords_trans_df.sort_values("Number of documents", ascending=False)
            my_output(self, "keywords_trans_df", f_name="\\tables\\" + f_name + " transformed.xlsx")
        my_output(self, "keywords_assoc_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def to_excel(self, f_name="report.xlsx", index=0, exclude=[], 
                 autofit=True, M_len=100):
        reportbib.results_to_excel(self, f_name=self.res_folder+"\\reports\\"+f_name, 
                                  index=index, exclude=exclude, 
                                  autofit=autofit, M_len=M_len)
        