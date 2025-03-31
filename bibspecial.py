# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:56:52 2023

@author: Lan.Umek
"""

import biblium
import readbib
import pandas as pd
import miscbib
import utilsbib
import plotbib

import sdg_queries

def get_sdg(df, var):
    for x, y in sdg_queries.pairs:
        df[y] = df[var].str.contains(x, case=False, regex=True, na=False).astype(int)
    try:
        df["life"] = df[["SDG 01", "SDG 02", "SDG 03"]].max(axis=1)
        df["economic and technological development"] = df[["SDG 08", "SDG 09"]].max(axis=1)
        
        df["social development"] = df[["SDG 11", "SDG 16"]].max(axis=1)
        df["equality"] = df[["SDG 04", "SDG 05", "SDG 10"]].max(axis=1)
        
        df["resources"] = df[["SDG 06", "SDG 07", "SDG 12"]].max(axis=1)
        df["natural environment"] = df[["SDG 13", "SDG 14", "SDG 15"]].max(axis=1)
        
        df["economic dimension"] = df[["life", "economic and technological development"]].max(axis=1)
        df["social dimension"] = df[["social development", "equality"]].max(axis=1)
        df["environmental dimension"] = df[["resources", "natural environment"]].max(axis=1)
    except:
        pass
    
    
    return df


class SDG_Analysis(biblium.BiblioAnalysis):
    
    def __init__(self, *args, **kwargs):

       super().__init__(*args, **kwargs)
            
       
       self.sdg_metadata = pd.read_excel(self.fd + "\\additional files\\Scopus_SDG_metadata.xlsx", sheet_name="goals")
       
       self.Props = self.sdg_metadata["number of documents"]/self.sdg_metadata["number of documents"].sum()
       self.Props.index = self.sdg_metadata["goal"]
       
       ns = self.df[[c for c in self.df.columns if "SDG" in c]].sum()
       self.props = ns/ns.sum()
       
       self.props_df = utilsbib.get_props_df(self.Props, self.props)
       
       add = self.sdg_metadata[["goal", "perspective", "name", "dimension"]].set_index("goal")
       self.props_df = pd.concat([self.props_df, add], axis=1, sort=True)
       
       
       self.color_dict_pers = {"life": "darkblue", "resources": "darkred", "equality": "lightgreen",
                         "economic and technological development": "lightblue", "social development": "darkgreen",
                         "natural environment": "lightcoral"}

       self.color_dict_dim = {"economic": "blue", "social": "green", "environmental": "red"}
       
    
    #def set_goals(self, var="Abstract"): # To sicer dela, a bo treba sistematično urediti
    #    self.df = get_sdg(self.df, var)
    
    def set_color_dicts(self, color_dict_pers=None, color_dict_dim=None):
       if color_dict_pers is not None:
           self.color_dict_pers = color_dict_pers
       if color_dict_dim is not None:
           self.color_dict_dim = color_dict_dim
           
           
    def plot_props_df(self, v="PP difference", pos_color="lightblue", neg_color="lightcoral",
                      name_var="name", x_label=None, 
                      show_0_line=True, p_lines=[], color_dict=None, color_var=None, 
                      padding = (20,20), show_col_leg=True, leg_loc="upper center", bbox_to_anchor=(0.3, -0.2),
                      f_name=None, dpi=1200, **kwds):
        
        plotbib.plot_props_df(self.props_df, v=v, pos_color=pos_color, neg_color=neg_color,
                          name_var=name_var, x_label=x_label, show_0_line=show_0_line, p_lines=p_lines, color_dict=self.color_dict_pers, color_var=None, 
                          padding=padding, show_col_leg=False, leg_loc=leg_loc, bbox_to_anchor=bbox_to_anchor, leg_n_col=2,
                          f_name=self.res_folder+f"\\plots\\SDG {v}", dpi=self.dpi, **kwds)
        
        plotbib.plot_props_df(self.props_df, v=v, pos_color=pos_color, neg_color=neg_color,
                          name_var=name_var, x_label=x_label, show_0_line=show_0_line, p_lines=p_lines, color_dict=self.color_dict_pers, color_var="perspective", 
                          padding=padding, show_col_leg=show_col_leg, leg_loc=leg_loc, bbox_to_anchor=bbox_to_anchor, leg_n_col=3,
                          f_name=self.res_folder+f"\\plots\\SDG by perspective {v}", dpi=self.dpi, **kwds)
        
        plotbib.plot_props_df(self.props_df, v=v, pos_color=pos_color, neg_color=neg_color,
                          name_var=name_var, x_label=x_label, show_0_line=show_0_line, p_lines=p_lines, color_dict=self.color_dict_dim, color_var="dimension", 
                          padding=padding, show_col_leg=show_col_leg, leg_loc=leg_loc, bbox_to_anchor=bbox_to_anchor, leg_n_col=1,
                          f_name=self.res_folder+f"\\plots\\SDG by dimension {v}", dpi=self.dpi, **kwds)
        
        """
        # tole so absolutni % po ciljih - razmisliti, če je ustrezno + barve se prekrivajo
        plotbib.plot_props_df(self.props_df, v="prop %", pos_color="blue", neg_color=neg_color,
                          name_var=name_var, x_label=x_label, show_0_line=show_0_line, p_lines=p_lines, color_dict=None, color_var=None, 
                          padding=(0, padding[1]), show_col_leg=show_col_leg, leg_loc=leg_loc, bbox_to_anchor=bbox_to_anchor, leg_n_col=1,
                          f_name=self.res_folder+"\\plots\\SDG by percentage", dpi=self.dpi, **kwds)
        
        plotbib.plot_props_df(self.props_df, v="prop %", pos_color="blue", neg_color=neg_color,
                          name_var=name_var, x_label=x_label, show_0_line=show_0_line, p_lines=p_lines, color_dict=self.color_dict_dim, color_var="dimension", 
                          padding=(0, padding[1]), show_col_leg=show_col_leg, leg_loc=leg_loc, bbox_to_anchor=bbox_to_anchor, leg_n_col=1,
                          f_name=self.res_folder+"\\plots\\SDG by percentage by dimension", dpi=self.dpi, **kwds)
        
        plotbib.plot_props_df(self.props_df, v="prop %", pos_color="blue", neg_color=neg_color,
                          name_var=name_var, x_label=x_label, show_0_line=show_0_line, p_lines=p_lines, color_dict=self.color_dict_pers, color_var="perspective", 
                          padding=(0, padding[1]), show_col_leg=show_col_leg, leg_loc=leg_loc, bbox_to_anchor=bbox_to_anchor, leg_n_col=1,
                          f_name=self.res_folder+"\\plots\\SDG by percentage by perspective", dpi=self.dpi, **kwds)
        """
        

           
class SDG_GroupAnalysis(biblium.BiblioGroup):
    
    def __init__(self, f_name=None, db="", df=None, bib_file=None, split_var=None, ind_df=None, c_var="topic", value_order=None,
                 res_folder="results - groups", save_results=True, preprocess=0,
                 norm_method = "jaccard", measures=["jaccard", "yule_Q"], 
                  adjust_p="fdr_bh", alpha=0.1, verbose=3, **kwds):
        
        self.db = db.lower()
        self.df = miscbib.read_data(f_name=f_name, db=self.db, df=df)

        ind_df = self.df[[c for c in self.df.columns if "SDG" in c]]
        
        super().__init__(f_name=f_name, db=db, df=df, bib_file=bib_file, 
                         split_var=None, ind_df=ind_df, value_order=value_order,
                         res_folder="results - groups - SDG goals", save_results=save_results,
                         preprocess=preprocess, norm_method=norm_method,
                         measures=measures, adjust_p=adjust_p, alpha=alpha, verbose=verbose)
        self.sdg_metadata = pd.read_excel(self.ba.fd + "\\additional files\\Scopus_SDG_metadata.xlsx", sheet_name="goals")


class SDG_GroupAnalysis_Perspective(biblium.BiblioGroup):
    
    def __init__(self, f_name=None, db="", df=None, bib_file=None, split_var=None, ind_df=None, value_order=None,
                 res_folder="results - groups", save_results=True, preprocess=0,
                 norm_method = "jaccard", measures=["jaccard", "yule_Q"], 
                  adjust_p="fdr_bh", alpha=0.1, verbose=3, **kwds):
        pers = ["life", "economic and technological development", "social development",
               "equality", "resources", "natural environment"]
        
        self.db = db.lower()
        self.df = miscbib.read_data(f_name=f_name, db=self.db, df=df)
                
        ind_df = self.df[[c for c in self.df.columns if c in pers]]
    
        super().__init__(f_name=f_name, db=db, df=df, bib_file=bib_file, 
                         split_var=None, ind_df=ind_df, value_order=value_order,
                         res_folder="results - groups - SDG perspectives", save_results=save_results,
                         preprocess=preprocess, norm_method=norm_method,
                         measures=measures, adjust_p=adjust_p, alpha=alpha, verbose=verbose)

class SDG_GroupAnalysis_Dimension(biblium.PlotBiblioGroup):
    
    def __init__(self, f_name=None, db="", df=None, bib_file=None, split_var=None, ind_df=None, value_order=None,
                 res_folder="results - groups", save_results=True, preprocess=0,
                 norm_method = "jaccard", measures=["jaccard", "yule_Q"], 
                  adjust_p="fdr_bh", alpha=0.1, verbose=3, **kwds):
        
        self.db = db.lower()
        self.df = miscbib.read_data(f_name=f_name, db=self.db, df=df)
                
        ind_df = self.df[[c for c in self.df.columns if "dimension" in c]]
    
        super().__init__(f_name=f_name, db=db, df=df, bib_file=bib_file, 
                         split_var=None, ind_df=ind_df, value_order=value_order,
                         res_folder="results - groups - SDG dimensions", save_results=save_results,
                         preprocess=preprocess, norm_method=norm_method,
                         measures=measures, adjust_p=adjust_p, alpha=alpha, verbose=verbose)

class Sciences_Analysis(biblium.BiblioAnalysis):
    pass

class Sciences_GroupAnalysis(biblium.BiblioAnalysis):
    pass