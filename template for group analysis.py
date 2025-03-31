# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:11:31 2023

@author: Lan
"""

import biblium, utilsbib, miscbib
import pandas as pd

f_name_data = "data\\dataset 04.xlsx"
f_name_terms = None

df = pd.read_excel(f_name_data)
if f_name_terms is not None:
    df_terms = pd.read_excel(f_name_terms)

v = "Authors Keywords"    
replacer_d={}
remove=[]
kwds={}

# Osnovna analiza

bg = biblium.BiblioGroupAnalysis(f_name=None, database="", df=None, bib_file=None, split_var=None, ind_df=None, value_order=None,
                 res_folder="results - groups", save_results=True,
                 norm_method = "jaccard",
                  measures=["jaccard", "yule_Q"], # dodaj cond_x in cond_y
                  adjust_p="fdr_bh", alpha=0.1, verbose=3)

bg.plot_overlapping(label_cbar=None, kind="heatmap", subset=None, include_totals=True, **kwds)

bg.get_main_info(f_name="main info")

bg.get_production(fill_empty=True, rng=None, cut_year=None, exclude_last="smart", f_name="scientific group production")

bg.plot_production(vrs1=None, vrs2=None, title=None, colors=None)

bg.get_top_cited_docs(top=10, where="global", limit_to=None, add_vars=[], rm_vars=[], f_name="top cited documents")

bg.count_sources(f_name="sources counts")

bg.get_sources_stats(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top sources stats")

bg.associate_sources(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True,
                          abbreviate=True, f_name="associated sources")

bg.get_authors_stats(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top authors stats")

bg.associate_authors(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True,
                          f_name="associated authors")

bg.get_ca_countries_stats(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top ca countries stats")

bg.associate_ca_countries(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True,
                          f_name="associated ca countries")

bg.get_keywords_stats(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=False,
                          abbreviate=True, f_name="top ca countries stats")

bg.associate_keywords(items=[], exclude=[], top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True,
                          f_name="associated keywords")

bg.plot_assoc_keywords(freq_var="a", color_var="yule_Q", f_name="wordcloud")

bg.tree_plot_assic_keywords(freq_var="a", color_var="yule_Q", f_name="treemap", show=None, font_size=10)

bg.set_local_analysis()

bg.to_excel(f_name="report.xlsx", index=0, exclude=[], autofit=True, M_len=100)



