# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:11:31 2023

@author: Lan
"""

import biblium, utilsbib, miscbib
import datetime
import numpy as np
import pandas as pd
import gensim
import networkx as nx
from cdlib import algorithms

# Tole je treba določiti

f_name_data = "data\\scopus.csv"
f_name_terms = None

df = pd.read_csv(f_name_data)
if f_name_terms is not None:
    df_terms = pd.read_excel(f_name_terms)
    
v = "Authors Keywords"
replacer_d={}
remove=[]
kwds={}

# Osnovna analiza

ba = biblium.BiblioAnalysis(f_name=None, database="scopus", df=df, bib_file=None, 
                 preprocess=1, save_results=True,
                 res_folder="results", default_keywords="ak", keep_orig=False,
                 lang_output="english", lang_docs="english", dpi_plots=600, 
                 sig_digits=2, verbose=2, color_plot="black", shade_col="gray", 
                 cmap="viridis", d_cmap="Set1")


#ba.show_data(sample=True, n=20)

ba.compute_add_features()

#ba.preprocess_abstract(lf=utilsbib.lf, exclude=["nan"], min_len=2, force=0)

#ba.add_stats_from_refs(min_year=1950, max_year=datetime.datetime.now().year,
#                        stats=[("Average year from refs", np.mean), 
#                               ("Number of references", len),
#                               ("Span of years from refs", miscbib.v_range)])

#ba.compute_variables_terms(v, df_terms, search="substring", cond="|")

ba.add_sources_data_scopus(f_name="sources data.xlsx", s_name="Scopus Sources May 2023", **kwds)

#ba.compute_misc_features()

#ba.filter_documents(*kwds)

ba.get_main_info(exclude=[], collab_index=True, spectroscopy=True, min_year=1950, f_name="main info")

ba.preprocess_keywords(which="aik", lf=utilsbib.lf, replacer_d=replacer_d, remove=remove)

ba.get_top_cited_docs(top=10, where="global", limit_to=None, 
                           add_vars=[], rm_vars=[], f_name="top cited documents")

#ba.get_top_cited_references(top=10, f_name="top cited references")

ba.get_production(fill_empty=True, rng=None, cut_year=None, exclude_last="smart", f_name="scientific production")

ba.plot_production(years=None, first_axis="Number of documents", 
                    second_axis="Cummulative number of citations", title=None,
                    comma=False, **kwds)

# štetje

ba.count_sources(f_name="sources counts")
ba.count_doc_types(f_name="document types counts")
ba.count_ca_countries(f_name="corresponding author country counts")
ba.count_authors(f_name="author counts")
#ba.count_areas(f_name="")
ba.count_keywords(which="ak", f_name="keyword counts")
#ba.count_references(f_name="references counts")
#ba.count_words_and_phrases(max_n=2,  post_clean=True, f_name="words and phrases counts") # tole za zdaj dela samo na povzetkih
#ba.count_local_cited_authors(f_name="local cited authors coutns")
#ba.count_local_cited_sources(f_name="local cited sources coutns")


# naprednejše statistike
ba.get_sources_stats(items=[], exclude=["nan"], top=20, top_by="Number of documents", min_freq=2, cov_prop=0.8, level=1, compute_indicators=True, abbreviate=True, f_name="top sources stats")

#ba.bar_plot_top_sources(by="Number of documents", top=10, name="short", core=False, c_var="Total number of citations", **kwds)

#ba.box_plot_top_sources(y_var="Cited by", items=[], exclude=[], top=5, boxtype="box", rotate=90, fontsize=10, core=False, **kwds)

ba.scatter_plot_top_sources(items=[], exclude=["nan"], top=10, min_freq=1, 
                                 by="Number of documents", x="Number of documents", y="Total number of citations",
                                 s="H-index", c="Average year of publication", l="Abbreviated Source Title",
                                 max_size=100, x_scale="log", y_scale="log", max_text_len=30,
                                 show_mean_line=False, show_mu=False, show_reg_line=False,
                                 arrowprops=None, core=True,                                
                                 kwds1={}, kwds2={})

ba.get_ca_countries_stats(items=[], exclude=["nan"], top=20, top_by="Number of documents", min_freq=5, cov_prop=0.8, level=1, compute_indicators=True, f_name="top ca countries stats")

#ba.bar_plot_top_ca_countries(by="Number of documents", top=10, name_var="CA Country", core=False, c_var="Total number of citations", **kwds)

ba.scatter_plot_top_ca_countries(items=[], exclude=["nan"], top=10, min_freq=1, 
                                 by="Number of documents", x="Number of documents", y="Total number of citations",
                                 s="H-index", c="Average year of publication", l="CA Country",
                                 max_size=100, x_scale="log", y_scale="log", 
                                 show_mean_line=False, show_mu=False, show_reg_line=False,
                                 arrowprops=None, core=True,                                
                                 kwds1={}, kwds2={})

ba.get_authors_stats(items=[], exclude=["nan"], top=20, top_by="Number of documents", 
                          min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True, f_name="top authors stats")

#ba.bar_plot_top_authors(by="Number of documents", top=10, name_var="Author", core=False, c_var="Total number of citations", **kwds)

#ba.box_plot_top_authors(y_var="Cited by", items=[], exclude=["nan"], top=5, boxtype="box", rotate=90, fontsize=10, core=False, **kwds)

ba.get_keywords_stats(items=[], exclude=["nan"], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True, f_name="top keywords stats")

#ba.get_fields_stats(items=[], exclude=["nan"], 
#                            top=20, top_by="Number of documents",
#                           min_freq=5, cov_prop=0.8, 
#                          freq_col="Number of documents", level=1,
#                          compute_indicators=True, f_name="top words and phrases stats")

#ba.get_words_and_phrases_stats(max_n=2, items=[], exclude=["nan"], 
#                            top=20, top_by="Number of documents",
#                           min_freq=5, cov_prop=0.8, 
#                          freq_col="f", level=1, f_name="top words and phrases stats")

#ba.get_refs_stats(items=[], exclude=["nan"], 
#                       top=20, top_by="Number of documents",
#                           min_freq=5, cov_prop=0.8, 
#                           freq_col="Number of documents", level=1,
#                           compute_indicators=True, f_name="top refs stats")


ba.plot_distr_citations(v="Cited by", min_cist=1, bins=None, nd=None, xlabel="Number of citations", ylabel="Number of documents", title=None, core=False, **kwds)

# dynamics

ba.get_sources_dynamics(items=None, max_items=20, fill_empty=True, f_name="source dynamics")

ba.plot_source_dynamics(items=[], max_items=5, xlabel="Year", ylabel="Number of documments", title=None, cum=False, years=None, l_title="Source title", core=False, **kwds)

ba.get_ca_country_dynamics(items=None, max_items=20, fill_empty=True, f_name="country dynamics")

ba.plot_top_authors_production(items=[], exclude=["nan"],
                                    y="Cited by", top=10, names=None, 
                                    xlabel="Year", ylabel="Author",
                                    size_one=20, label="Number of citations", 
                                    core=False)

ba.plot_top_ca_countries_production(items=[], exclude=["nan"],
                                    y="Cited by", top=10, names=None, 
                                    xlabel="Year", ylabel="Country of corresponding author",
                                    size_one=20, label="Number of citations", 
                                    core=False)


ba.get_country_collaboration_df(f_name="country collaboration")

ba.plot_country_collab_production(s_var="SCP", m_var="MCP", l_var="Country of corresponding author", top=10, f_name="country collaboration production", core=False, **kwds)

#ba.mds_df_from_aspects(fields=["keywords", "sources", "authors"], 
#                            ks=10, customs=[()], nc=2,  metric="cosine",
#                            rename_dct_group={}, f_name="mds coordinates") # dodati bi bilo treba funkcijo za izris

#ba.scatter_plot_mds(x=0, y=1, f_name="mds plot") # preveri, če je to ta

#ba.plot_spectrosopy(core=False, **kwds)

#ba.scatter_plot_ref_age(year_range=None)

#ba.scatter_plot_ref_age_density(year_range=None, avg=True)

#ba.scatter_plot_citations_citescore(x="CiteScore", y="Cited by", s=None, c="Year", l=None, show_id_line=True)


ba.trend_topics(what="ak", items=[], exclude=["nan"], name_var=None, period=None, 
                     left_var="Q1 year", right_var="Q3 year", mid_var="median year", 
                     s_var="Number of documents", c_var="Total number of citations",
                     items_per_year=5, x_label="Year (Q1-median-Q3)")

#ba.plot_word_cloud(by="abstracts", name_var=None, freq_var=None, color_var="Average year of publication",  background_color="white", f_name="wordcloud", dpi=600, **kwds)

# sopojavitve

ba.get_kw_co_network(normalize="association", 
                     sort_matrix=True, remove_trivial=False, **kwds) # preveri, kaj je s tem (verjetno je odveč in zastarel)

ba.get_keyword_co_net(df_ind=None, which="ak", items=[], exclude=["nan"], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, freq_col="Number of documents",
                           norm_method="association",
                           partition_algorithm=algorithms.walktrap,
                           f_name="keyword co-occurrence network",
                           save_vectors=["Number of documents", "Average year of publication", "H-index"])

ba.plot_keyword_co_net(color_var="Average year of publication",
                        size_var="Number of documents", layout=nx.spring_layout, adjust_labels=True, **kwds)

#ba.plot_keyword_co_heatmap(**kwds)

ba.thematic_plot_co_keyword(k=None, rnk=True, s_var="Number of documents", c_var="Average year")

#ba.get_co_authorship_network(normalize="association", 
#                                  sort_matrix=True, remove_trivial=False,
#                                  **kwds) # to je tudi verejtno zastarelo

#ba.get_cocitation_net(df_ind=None, items=[], exclude=["nan"],top=20, top_by="Number of documents",
#                           min_freq=5, cov_prop=0.8, freq_col="Number of local citations",
#                           norm_method="association",   partition_algorithm=algorithms.walktrap,
#                           f_name="co-citation network", save_vectors=["Number of documents", "Average year of publication", "H-index"])

ba.get_topics(text_var="Abstract", model=gensim.models.LdaModel,
                   num_topics=10, stop_words=utilsbib.add_stopwords, show_words_per_topic=100, f_name="topics (topic modelling) ") # tole je treba dodelati še za izrsi

#ba.plot_loadings(c1=0, c2=1, min_norm=0.1, col1="r", col2="b", 
 #                     l1=None, l2=None, f_name="loadings", show_grid=True)

ba.k_fields_plot(fields=["keywords", "sources"], customs=[], ks=10, 
                      color="Average year", add_colorbar=False, save_html=False, font_size=10)

#ba.plot_tree_map(field="keywords", size_var="Number of documents", color_var="Average year of publication",
#                 labels_var=None, show=None, font_size=10)

# reporting 

ba.to_excel(f_name="report.xlsx", index=0, exclude=[], 
                 autofit=True, M_len=100)

ba.to_word(f_name="report.docx", include_non_core=False) # manjka še to_latex