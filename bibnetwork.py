# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:25:58 2023

@author: Lan
"""

from bibstats import BiblioStats
from cdlib import algorithms
import miscbib


class BiblioNetwork(BiblioStats):
    
    def get_keyword_co_net(self, df_ind=None, which="ak", items=[], exclude=[], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, freq_col="Number of documents",
                           norm_method="association",
                           partition_algorithm=algorithms.louvain,
                           f_name="keyword co-occurrence network",
                           save_vectors=["Number of documents", "Average year of publication", "H-index"]):
        
        if df_ind is None:
            self.count_keywords()
            self.get_keywords_stats(which=which, items=items, exclude=exclude, 
                        top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop, 
                        freq_col="Number of documents", compute_indicators=True)
            df_ind = self.keywords_df_ind
        self.KW_cooc_net = miscbib.BibNetwork(df_ind, norm_method=norm_method, partition_algorithm=partition_algorithm)
        self.KW_cooc_net.get_vectors(self.keywords_df)
        
        if f_name is not None:
            f_name = self.res_folder +"\\networks\\" + f_name
            self.KW_cooc_net.to_pajek(f_name, save_vectors=save_vectors)
            
            
    def get_fileds_co_net(self, df_ind=None,items=[], exclude=[], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, freq_col="Number of documents",
                           norm_method="association",
                           partition_algorithm=algorithms.louvain,
                           f_name="fields co-occurrence network",
                           save_vectors=["Number of documents", "Average year of publication", "H-index"]):
        
        if df_ind is None:
            self.count_areas()
            self.get_fields_stats(items=items, exclude=exclude, 
                        top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop, 
                        freq_col="Number of documents", compute_indicators=True)
            df_ind = self.fields_df_ind
        self.F_cooc_net = miscbib.BibNetwork(df_ind, norm_method=norm_method, partition_algorithm=algorithms.louvain) # tukaj ga prisili, da da Science za barvo
        self.F_cooc_net.get_vectors(self.fields_df)
        
        if f_name is not None:
            f_name = self.res_folder +"\\networks\\" + f_name
            self.F_cooc_net.to_pajek(f_name, save_vectors=save_vectors)
            
    def get_cocitation_net(self, df_ind=None, items=[], exclude=[],top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, freq_col="Number of local citations",
                           norm_method="association",   partition_algorithm=algorithms.louvain,
                           f_name="co-citation network", save_vectors=["Number of documents", "Average year of publication", "H-index"]):
        if df_ind is None:
            if not hasattr(self, "refs_df_ind"):
                self.count_references(top=top)

            df_ind = self.refs_df_ind
        rename_dict = dict(zip(self.refs_df["Reference"], self.refs_df["Short name"]))
        self.cocitation_net = miscbib.BibNetwork(df_ind, norm_method=norm_method, partition_algorithm=partition_algorithm, rename_dict=rename_dict)
        self.cocitation_net.get_vectors(self.refs_df)
            
        if f_name is not None:
            f_name = self.res_folder +"\\networks\\" + f_name
            self.cocitation_net.to_pajek(f_name, save_vectors=save_vectors)
        
            
