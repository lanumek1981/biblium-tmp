# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:14:13 2023

@author: Lan
"""


import pandas as pd
import numpy as np
import os
import readbib, miscbib, utilsbib, plotbib, reportbib
import datetime
import gensim

class BiblioStats:
    
    def __init__(self, f_name=None, database="", df=None, bib_file=None, 
                 preprocess=1, save_results=True,
                 res_folder="results", default_keywords="ak",
                 lang_output="english", lang_docs="english", dpi_plots=600, 
                 sig_digits=2, verbose=2, color_plot="black", shade_col="gray", 
                 cmap="viridis", d_cmap="Set1"):
        
        self.database = database.lower()
        if f_name is not None:
            self.df = readbib.read_bibfile(f_name, self.database)
        elif df is not None:
            self.df = df
        elif bib_file is not None:
            self.df, self.database = bib_file
        else:
            print("No input file given")
            return None     
        if self.database not in ["scopus", "wos"]:
            print("Warning")
            return None


        self.n = len(self.df)
        self.fd = os.path.dirname(__file__)
        
        if self.database == "scopus":
            self.df_scopus_codes = pd.read_excel(self.fd + "\\additional files\\scopus subject area codes.xlsx")
        
        self.save_results = save_results
        sub_folders = ["plots", "tables", "reports", "indicators", "networks"] 
        self.res_folder = os.getcwd() + "\\" + res_folder
        if self.save_results:
            folders = [self.res_folder + "\\" + s for s in sub_folders]
            utilsbib.make_folders(folders)
        
        self.images = []    
        self.dpi, self.sig_digits = dpi_plots, sig_digits
        self.color_plot, self.shade_col = color_plot, shade_col
        self.cmap, self.d_cmap = cmap, d_cmap
        
        self.verbose = verbose
        
        self.lang_output, self.lang_docs = lang_output, lang_docs
        
        self.current_year = datetime.datetime.now().year
        if preprocess >= 3:
            self.preprocess_abstract()
        if preprocess >= 2:
            self.df = miscbib.drop_irrelevant_features(self.df)
        if preprocess >= 1:
            self.compute_add_features()
            self.get_mappings()
        else:
            self.dct_sources_abb = None
            
        self.kw_var = miscbib.which_keywords(self.df, which=default_keywords)
        
    def show_data(self, sample=True, n=20):
        utilsbib.make_folders([self.res_folder + "\\sample data"])
        f_name = self.res_folder + "\\sample data\\working data.xlsx"
        if sample:
            df = self.df.sample(n=n)
            f_name = self.res_folder + "\\sample data\\working data sample.xlsx"
        else:
            df = self.df
            f_name = self.res_folder + "\\sample data\\working data.xlsx"
        df.to_excel(f_name, index=False)
        os.startfile(f_name)

        
    # preprocessing
    
    def compute_add_features(self):
        self.df = miscbib.get_cits_per_year(self.df, self.current_year)
        self.df = miscbib.get_doc_names(self.df)
        self.df = miscbib.add_ca_country_df(self.df, self.database)
        self.df, self.list_countries = miscbib.extract_affs_and_countries(self.df, self.database)
        self.get_country_collaboration_df()
        self.df = miscbib.get_topic_var(self.df)
        
        
        
    def get_mappings(self):
        self.dct_sources_abb = miscbib.get_sources_abb_dict(self.df)
        self.dct_au = miscbib.get_dict_of_authors(self.df, database=self.database)
        self.dct_au_reverse = {v: k for k, v in self.dct_au.items()}
        
    def preprocess_abstract(self, lf=utilsbib.lf, exclude=[], min_len=2, force=0):
        if "abstract" in self.df.columns and force != 1:
            print("Abstract already preprocessed. If you intend to (re)preprocess,\n", 
                  "call the function with force=1")
            return None
        self.df["abstract"] = self.df["Abstract"].map(lambda x: utilsbib.lemmatize_text(
                x, lf=lf, sw_l=self.lang_docs, exclude=exclude, min_len=min_len)) 
        self.df["words from abstract"] = self.df["abstract"].map(lambda x: utilsbib.count_un_words(x,
               sw_l=self.lang_docs, exclude=exclude, min_len=min_len))
        
    def add_stats_from_refs(self, min_year=1950, max_year=datetime.datetime.now().year,
                        stats=[("Average year from refs", np.mean), 
                               ("Number of references", len),
                               ("Span of years from refs", miscbib.v_range)]):
        self.df, self.ref_years_0 = miscbib.add_stats_from_refs(
            self.df, min_year=min_year, max_year=max_year, stats=stats)
        self.ref_years = utilsbib.unnest(self.ref_years_0)
        doc_ref_pairs = []
        for i in range(self.n):
            if len(self.ref_years_0[i]):
                doc_ref_pairs += [(self.df["Year"].iloc[i], j) for j in self.ref_years_0[i]]
        self.df_doc_ref_pairs = pd.DataFrame(doc_ref_pairs, columns=["Year", "Year of reference"])
        
    # additional features based on words that link to terms
    
    def compute_variables_terms(self, v, df_terms, search="substring", cond="|"):
        # df_terms - prvi stolpec pojmi, drugi navezave za pojme
        d_terms = utilsbib.link_terms(df_terms)
        self.new_vars_terms = list(d_terms.keys())
        self.df = utilsbib.compute_variables(
            self.df, v, list(d_terms.values()), 
            names=self.new_vars_terms, search=search, cond=cond)
        
    def add_sources_data_scopus(self, f_name="sources data.xlsx", s_name="Scopus Sources May 2022", **kwds):
        if self.database != "scopus":
            print("Works for Scopus database only")
            return None

        if "Codes" not in self.df.columns:
            try:
                df_sources = pd.read_excel(self.fd+"\\additional files\\" + f_name, sheet_name=s_name)
            except:
                print("Dowload source info file from Scopus webpage")
                return None
            self.df = pd.concat([self.df, miscbib.add_source_info(self.df, df_sources, **kwds)], axis=1)
            self.df["Citations to CiteScore diffrence"] = self.df["Cited by"] - self.df["CiteScore"]
            self.df["Citations to CiteScore ratio"] = self.df["Cited by"] / self.df["CiteScore"]
        self.df = miscbib.translate_scopus_codes(self.df)
        
    # various computed features
    
    def compute_misc_features(self):
        if "Title" in self.df.columns or "title" in self.df.columns:
            t_var = "title" if "title" in self.df.columns else "Title"
            self.df["Number of words in title"] = self.df[t_var].map(utilsbib.count_words_string)
            punc = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
            cn = lambda l: sum([1 for x in l if x in punc ])
            if "Title" in self.df.columns: # title (with small t) can be without punctuations
                self.df["Number of punctuations in title"] = self.df["Title"].map(cn)
        if "Abstract" in self.df.columns or "abstract" in self.df.columns:
            a_var = "abstract" if "abstract" in self.df.columns else "Abstract"
            self.df["Number of words in abstract"] = self.df[a_var].map(utilsbib.count_words_string)
        self.df = miscbib.count_self_ref_sources(self.df)
    
    # filtering
    
    def filter_documents(self, **kwds):
        # keyword arguments: year_range=None, min_citations=-1, include_types=[], exclude_types=[], include_keywords=[], exclude_keywords=[], include_in_topic=[], exclude_from_topic=[], topic_var="abstract", languages=[], include_disciplines=[], exclude_disciplines=[], include_sources=[], exclude_sources=[], discipline_var="Scopus Sub-Subject Area"
        self.df = miscbib.filter_documents(self.df, self.database, verbose=self.verbose, **kwds)
        
    # decsriptives
        
    def get_main_info(self, exclude=[], collab_index=True, spectroscopy=True, min_year=1950):
        self.gl_perf_dict, self.ref_stats, self.ak_stats = {}, {}, {}
        vrs = [c for c in self.df.columns if c not in exclude]
        if "Year" in vrs:
            self.year_stats = miscbib.year_stats_df(self.df)
        if "Source title" in vrs:
            if not hasattr(self, "n_sources"):
                self.n_sources = len(self.df["Source title"].unique())
        if "Cited by" in vrs:
            self.cits_stats = miscbib.cits_stats_df(self.df)
        self.df = self.df.replace({pd.NA: np.nan})
        if "References" in vrs:
            self.ref_stats = miscbib.occ_stats_df(self.df, "References", sep=";")
            if spectroscopy:
                if not hasattr(self, "ref_years"):
                    self.add_stats_from_refs(min_year=min_year)
                self.ref_stats["Average year of reference"] = np.mean(self.ref_years)
                self.ref_stats["Correlation (publication year-avg reference year)"] = self.df[["Year", "Average year from refs"]].dropna(subset=["Year", "Average year from refs"]).corr()["Year"]["Average year from refs"]
        if "Author Keywords" in vrs:
            self.ak_stats = miscbib.occ_stats_df(self.df, "Author Keywords", sep="; ")
        if "Index Keywords" in vrs:
            self.ik_stats = miscbib.occ_stats_df(self.df, "Index Keywords", sep="; ")
        # poglej, kje bi dal [No author id available] kot dodatek za rm
        rm = ["", " ", "[No author id available]"]
        if "Author(s) ID" in vrs:
            self.au_stats = miscbib.occ_stats_df(self.df, "Author(s) ID", sep=";") # tule spremeni, da doaš rm
        elif "Authors" in vrs:
            self.au_stats = miscbib.occ_stats_df(self.df, "Authors", sep=", ")
        if collab_index:
            if "Author(s) ID" in vrs:
                self.au_stats.update(miscbib.collab_stats(self.df["Author(s) ID"], ";", rm=rm))
            elif "Authors" in vrs:
                self.au_stats.update(miscbib.collab_stats(self.df["Authors"], ", ", rm=rm))
        
        self.gl_perf_dict = dict(miscbib.get_perf_ind(self.df, level=3))
        if "H-index" in self.gl_perf_dict:
            self.df_h_score = self.df.sort_values(
                "Cited by", ascending=False).head(self.gl_perf_dict["H-index"])
        # dodaj statistiko o državah
        if hasattr(self, "country_collab_df"):
            self.int_collab_index = self.country_collab_df["MCP"].sum() / self.country_collab_df["Number of documents"].sum()
            self.int_collab_index = np.round(self.int_collab_index, self.sig_digits)
            self.au_stats.update({"International collboration index": self.int_collab_index})
        
        self.main_info_df = utilsbib.df_from_list_of_dcts([self.gl_perf_dict, self.au_stats, self.ref_stats, self.ak_stats])
        self.main_info_df["value"] = self.main_info_df["value"].map(lambda x: utilsbib.smart_round(x, ndig=self.sig_digits))
                
    def preprocess_keywords(self, which="aik", lf=utilsbib.lf, replacer_d={}, remove=[]): #
        if which in ["ak", "aik"]:
            self.df["author keywords"] = self.df["Author Keywords"].map(
                lambda x: utilsbib.preprocess_keywords(
                    x, sw_l=self.lang_docs, lf=lf, 
                    s_char="; ", replacer_d=replacer_d, remove=remove))
        if which in ["ik", "aik"]:
            self.df["index keywords"] = self.df["Index Keywords"].map(
                lambda x: utilsbib.preprocess_keywords(
                    x, sw_l=self.lang_docs, lf=lf, 
                    s_char="; ", replacer_d=replacer_d, remove=remove))
        # sestavi author in index keywords ter keywords
            
    def get_top_cited_docs(self, top=10, where="global", limit_to=None, 
                           add_vars=[], rm_vars=[]):
        if (where == "global") or (where == "both"):
            self.top_gl_cit_docs_df = miscbib.get_top_gl_cit_docs(
                self.df, top=top, limit_to=limit_to, add_vars=add_vars, rm_vars=rm_vars)
        if where == "local" or "both":
            print("TO DO")

    def get_production(self,  fill_empty=True, rng=None, cut_year=None, exclude_last="smart"):
        self.production_df = miscbib.get_production(
                self.df, fill_empty=fill_empty, rng=rng, cut_year=cut_year, exclude_last=exclude_last)
        
    # counting
    
    def count_sources(self):
        self.sources_df = utilsbib.freqs_from_col(self.df, "Source title",
                                                  rename_dct=miscbib.freqs_rename_dct)
        self.all_sources = self.sources_df["Source title"].tolist()
        self.n_sources = len(self.all_sources)
        if self.dct_sources_abb is not None:
            self.sources_df["Abbreviated Source Title"] = self.sources_df["Source title"].map(self.dct_sources_abb)
            
    def count_doc_types(self):
        self.doc_type_df = utilsbib.freqs_from_col(self.df, "Document Type",
                                                  rename_dct=miscbib.freqs_rename_dct)
        
    def count_ca_countries(self):
        if "CA Country" not in self.df.columns:
            if "Correspondence Address" in self.df.columns:
                self.df = miscbib.add_ca_country_df(self.df, self.database)            
        self.ca_countries_df = utilsbib.freqs_from_col(self.df, "CA Country",
                                                       rename_dct=miscbib.freqs_rename_dct)
        
    def count_authors(self, rm=["", " ", "[No author id available]"]):
        if "Author(s) ID" in self.df.columns:
            self.authors_df = miscbib.both_counts_occurrences(self.df["Author(s) ID"], ";", rm=rm)
            self.au_var = "Author(s) ID"
        elif "Authors" in self.df.columns:
            self.authors_df = miscbib.both_counts_occurrences(self.df["Authors"], ", ", rm=rm)
            self.au_var = "Authors"
        self.authors_df["Author"] = self.authors_df["item"].map(self.dct_au)
        self.authors_df = self.authors_df.rename(columns={"item": self.au_var, "count": "Number of documents"})
        # urediti vrstni red stolpcev
        
    def count_areas(self):
        if self.database == "scopus":
            if "Codes" not in self.df.columns:
                print("Codes not available")
                return None
            else:
                self.fields_df = miscbib.both_counts_occurrences(
                    self.df["Fields"], "; ").rename(
                        columns={"item": "Field", "count": "Number of documents"})
                self.areas_df = miscbib.both_counts_occurrences(
                    self.df["Areas"], "; ").rename(
                        columns={"item": "Area", "count": "Number of documents"})
                self.sciences_df = miscbib.both_counts_occurrences(
                    self.df["Sciences"], "; ").rename(
                        columns={"item": "Science", "count": "Number of documents"})
        else:
            print("Not supported")
        
    def count_keywords(self, which="ak"):
        self.kw_var = miscbib.which_keywords(self.df, which=which)
        self.keywords_df = miscbib.count_occurrences_df(self.df[self.kw_var], "; ")
        self.keywords_df = self.keywords_df.rename(columns={"item": "Keyword"})
        
    def count_references(self):
        self.refs_df = miscbib.count_occurrences_df(self.df["References"], "; ")
        self.refs_df = self.refs_df.rename(columns={"item": "Reference"})
        
    def count_words_and_phrases(self, max_n=2):
        self.ab_var = "abstract" if "abstract" in self.df.columns else "Abstract"
        self.abstracts = " ".join(self.df[self.ab_var])
        self.words_and_phrases = utilsbib.count_words_and_phrases(self.abstracts, max_n=max_n)
        self.words_and_phrases_df = pd.DataFrame(self.words_and_phrases, columns=["f", "word"])
        
    # more advanced statistics
    
    def get_sources_stats(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True,
                          abbreviate=True):
        if not hasattr(self, "sources_df"):
            self.count_sources()
        self.sources_df = miscbib.get_add_stats(
            self.df, self.sources_df, "Source title", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop)
        items = miscbib.select_items(
            self.sources_df, "Source title", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, freq_col="Number of documents", cov_prop=cov_prop)
        if compute_indicators:
            self.sources_df_ind = utilsbib.get_indicator_df(self.df, "Source title", items=items)
            if abbreviate:
                self.sources_df_ind = self.sources_df_ind.rename(columns=self.dct_sources_abb)
        
    def get_ca_countries_stats(self, items=[], exclude=[], 
                               top=20, top_by="Number of documents", 
                               min_freq=5, cov_prop=0.8, level=1, compute_indicators=True):
        # ideja: če daš items="eu" ali pa items="europe", ti izbere samo ustrezne države
        if type(items) == str:
            if items.lower() == "eu":
                items = miscbib.eu_countries
        if not hasattr(self, "ca_countries_df"):
            self.count_ca_countries()
        self.ca_countries_df = miscbib.get_add_stats(
            self.df, self.ca_countries_df, "CA Country", 
            items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop)
        items = miscbib.select_items(
            self.ca_countries_df, "CA Country", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, freq_col="Number of documents", cov_prop=cov_prop) # morda za eu ne dela ta del
        if compute_indicators:
            self.ca_countries_df_ind = utilsbib.get_indicator_df(self.df, "CA Country", items=items)   
        
    def get_authors_stats(self, items=[], exclude=[], 
                          top=20, top_by="Number of documents", 
                          min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True):
        if not hasattr(self, "authors_df"):
            self.count_authors()
        items = miscbib.select_items(
            self.authors_df, "Author", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq,
            freq_col=freq_col, cov_prop=cov_prop)
        items = [self.dct_au_reverse[it] for it in items]
        ps = miscbib.get_performances(self.df, "Author(s) ID", items, search="substring", level=level)
        self.authors_df = pd.merge(self.authors_df, ps, on="Author(s) ID")
        self.authors_df = miscbib.order_by_cits(self.authors_df)
        if compute_indicators:
            self.authors_df_ind = utilsbib.get_indicator_df(
                self.df, "Author(s) ID", search="substring", items=items)
            self.authors_df_ind = self.authors_df_ind.rename(columns=self.dct_au) 
        
    def get_keywords_stats(self, items=[], exclude=[], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True):
        if not hasattr(self, "keywords_df"):
            self.count_keywords()
        if type(items) == str:
            items = [k for k in self.keywords_df["Keyword"] if items in k]
        items = miscbib.select_items(
            self.keywords_df, "Keyword", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, 
            freq_col=freq_col, cov_prop=cov_prop)
        ps = miscbib.get_performances(self.df, self.kw_var, items, search="substring", level=level)
        ps = ps.rename(columns={self.kw_var: "Keyword"})
        self.keywords_df = pd.merge(self.keywords_df, ps, on="Keyword")
        self.keywords_df = miscbib.order_by_cits(self.keywords_df)
        if compute_indicators:
            self.keywords_df_ind = utilsbib.get_indicator_df(
                self.df, self.kw_var, search="substring", items=items)
            
    def get_fields_stats(self, items=[], exclude=[], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True):
        if not hasattr(self, "fields_df"):
            self.count_areas()
        #if type(items) == str:
        #    items = [k for k in self.keywords_df["Keyword"] if items in k]
        items = miscbib.select_items(
            self.fields_df, "Field", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, 
            freq_col=freq_col, cov_prop=cov_prop)
        ps = miscbib.get_performances(self.df, "Fields", items, search="substring", level=level)
        ps = ps.rename(columns={"Fields": "Field"})
        self.fields_df = pd.merge(self.fields_df, ps, on="Field")
        self.fields_df = miscbib.order_by_cits(self.fields_df)
        if compute_indicators:
            self.fields_df_ind = utilsbib.get_indicator_df(
                self.df, "Fields", search="substring", items=items)
    
    def get_words_and_phrases_stats(self, max_n=2, items=[], exclude=[], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="f", level=1):
        if not hasattr(self, "words_and_phrases_df"):
            self.count_words_and_phrases(max_n=max_n)
        items = miscbib.select_items(
            self.words_and_phrases_df, "word", items=items, exclude=exclude, 
            top=top, top_by="f", min_freq=min_freq, 
            freq_col=freq_col, cov_prop=cov_prop)
        ps = miscbib.get_performances(self.df, self.ab_var, items, search="substring", level=level)
        ps = ps.rename(columns={self.ab_var: "word"})
        self.words_and_phrases_df = pd.merge(self.words_and_phrases_df, ps, on="word")
        self.words_and_phrases_df = miscbib.order_by_cits(self.words_and_phrases_df)
           
    def get_refs_stats(self, items=[], exclude=[], 
                       top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                           freq_col="Number of documents", level=1,
                           compute_indicators=True):   
        # precej nerelevantna zadeva
        print("Just because you can do it, it does not mean that you should do it.")
        if not hasattr(self, "refs_df"):
            self.count_references()
        items = miscbib.select_items(
            self.refs_df, "Reference", items=items, exclude=exclude, 
            top=top, top_by="f", min_freq=min_freq, 
            freq_col=freq_col, cov_prop=cov_prop)
        ps = miscbib.get_performances(self.df, "References", items, search="substring", level=level)
        ps = ps.rename(columns={"References": "Reference"})
        self.refs_df = pd.merge(self.refs_df, ps, on="Reference")
        #self.refs_df = miscbib.order_by_cits(self.refs_df)
        if compute_indicators:
            self.refs_df_ind = utilsbib.get_indicator_df(
                self.df, "References", search="substring", items=items)
    
    # dynamics
    
    def get_sources_dynamics(self, items=None, max_items=20, fill_empty=True):
        self.ind_df_sources = utilsbib.get_indicator_df(self.df, "Source title", items=items, max_items=max_items)
        self.ind_df_sources = self.ind_df_sources.rename(columns=self.dct_sources_abb)
        self.sources_dyn_df = miscbib.get_productions_ind(
            self.ind_df_sources, self.df["Year"], fill_empty=fill_empty)
        
        
    def get_ca_country_dynamics(self, items=None, max_items=20, fill_empty=True):
        if items == "eu": # še ne dela OK
            items = miscbib.eu_countries
            max_items = 28
        self.ind_df_ca_countries = utilsbib.get_indicator_df(self.df, "CA Country", items=items, max_items=max_items)
        self.ca_country_dyn_df = miscbib.get_productions_ind(
            self.ind_df_ca_countries, self.df["Year"], fill_empty=fill_empty)
        
    def get_country_collaboration_df(self):
        self.country_collab_df = miscbib.get_country_collaboration_df(self.df)
        
    # combinations of different aspects
    
    def mds_df_from_aspects(self, fields=["keywords", "sources", "authors"], 
                            ks=10, customs=[()], nc=2,  metric="cosine",
                            rename_dct_group={}):
        # customs shoud be list of tuples (field, df_ind, k)
        dfs = []
        if ks is None:
            ks = [None] * len(fields)
        elif type(ks) == int:
            ks = [ks] * len(fields)
        for i, field in enumerate(fields):
            if hasattr(self, field+"_df_ind"):
                df_tmp = getattr(self,field+"_df_ind")
                if ks[i] is not None:               
                    dfs.append(df_tmp.iloc[:,:ks[i]])
                else:
                    dfs.append(df_tmp)
            else:
                print(f"Wrong input for {field}")
        for custom in customs:
            if len(custom) == 0:
                break
            field, df_ind, k = custom
            if k is not None:
                dfs.append(df_ind.iloc[:,:k])
            else:
                dfs.append(df_ind)
            fields.append(field)
        self.df_mds = miscbib.mds_from_dfs(dfs, groups=fields, nc=nc, metric=metric)
        self.df_mds["group"] = self.df_mds["group"].replace(rename_dct_group)

        
    #za avtorje ne dela OK
    #def get_authors_dynamics(self, items=None, max_items=20):
    #    self.ind_df_authors = utilsbib.get_indicator_df(self.df, "Authors", items=items, max_items=max_items, search="substring")
        
    # top authors, ca countires ... time production



    # co-occurrences
              
    def get_kw_co_network(self, normalize="association", 
                          sort_matrix=True, remove_trivial=False, **kwds):
        if not hasattr(self, "keywords_df_ind"):
            self.get_keywords_stats(**kwds)
        self.kw_co_mat = self.keywords_df_ind.T.dot(self.keywords_df_ind)
        self.kw_co_s_mat = utilsbib.normalize_sq_matrix(self.kw_co_mat, method=normalize)
        if sort_matrix:
            self.kw_sorted_mat = utilsbib.sort_matrix_elements(
                self.kw_co_mat, remove_trivial=remove_trivial)
            self.kw_sorted_s_mat = utilsbib.sort_matrix_elements(
                self.kw_co_s_mat, remove_trivial=remove_trivial) 
        
    def get_co_authorship_network(self, normalize="association", 
                                  sort_matrix=True, remove_trivial=False,
                                  **kwds):
        # tole popravi, da računa za vse avtorje, ne da bi zahteval sklic za izračun statistik
        if not hasattr(self, "authors_df_ind"):
            self.get_authors_stats(**kwds)
        self.co_auth_mat = self.authors_df_ind.T.dot(self.authors_df_ind)
        self.co_auth_mat = self.co_auth_mat.rename(index=self.dct_au, columns=self.dct_au)
        self.co_auth_s_mat = utilsbib.normalize_sq_matrix(self.co_auth_mat, method=normalize)
        if sort_matrix:
            self.co_auth_sorted_mat = utilsbib.sort_matrix_elements(
                self.co_auth_mat, remove_trivial=remove_trivial)
            self.co_auth_sorted_s_mat = utilsbib.sort_matrix_elements(
                self.co_auth_s_mat, remove_trivial=remove_trivial) 

    # topic modelling
    
    def get_topics(self, text_var="Abstract", model=gensim.models.LdaModel,
                   num_topics=10, stop_words=utilsbib.add_stopwords, show_words_per_topic=100):
        text_var = text_var.lower() if text_var.lower() in self.df.columns else text_var
        self.lda_model, self.doc_lda, self.id2word, self.corpus \
            = utilsbib.lda_topics(self.df, text_var, model=model, 
                                  num_topics=num_topics, stop_words=stop_words)
        self.top_topics = self.lda_model.top_topics(self.corpus, topn=show_words_per_topic)
        self.df_topics = pd.concat([pd.DataFrame(
            t[0], columns=["w %d " % (i+1) , "topic %d" % (i+1)])
            for i, t in enumerate(self.top_topics)], axis=1)
        
    
    # reporting
         
    def to_excel(self, f_name="report.xlsx", index=0, exclude=[], 
                 autofit=True, M_len=100):
        reportbib.results_to_excel(self, f_name=self.res_folder+"\\reports\\"+f_name, 
                                  index=index, exclude=exclude, 
                                  autofit=autofit, M_len=M_len)
        
    def to_word(self, f_name="report.docx", include_non_core=False):
        reportbib.results_to_word(self, f_name=self.res_folder+"\\reports\\"+f_name,
                                  include_non_core=include_non_core)
                                  
        
    def compute_all(self, level=3, plots=True, save_reports=True, **kwds):
        
        if level >= 1:
            self.get_main_info()
            self.get_top_cited_docs()
            self.get_production()
            if plots:
                self.plot_production()
        if level >= 2:
            self.get_sources_stats()
            self.get_ca_countries_stats()
            self.get_authors_stats()
            self.plot_country_collab_production()
            if plots:
                self.scatter_plot_top_sources()
                self.scatter_plot_top_ca_countries()
                
        if save_reports:
            self.to_excel()
            self.to_word()
