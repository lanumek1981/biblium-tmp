# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:14:13 2023

@author: Lan
"""


import pandas as pd
import numpy as np
import os
import readbib, miscbib, utilsbib, reportbib
import datetime
import gensim
from scipy.stats import entropy


def my_output(obj, att, f_name=None, additional=None, space="-"*20):
    if f_name is None:
        return None
    if obj.save_results and (obj is not None):
        f_name = obj.res_folder + f_name
        getattr(obj, att).to_excel(f_name, index=False)
    if obj.verbose > 1:
        print(att, "computed\n")
    if (obj.verbose > 2) and (obj.save_results):
        print(att, f" saved to {f_name}\n")
    if obj.verbose > 3:
        if additional is not None:
            print(f"Call {additional} for more stats\n")
    print(space+"\n")

class BiblioStats:
    
    def __init__(self, f_name=None, db="", df=None, bib_file=None,
                 pre_compute=1, drop_irrelevant=True,
                 preprocess=1, save_results=True, get_sciences=False,
                 res_folder="results", default_keywords="ak", keep_orig=False,
                 lang_output="english", lang_docs="english", dpi_plots=600, 
                 sig_digits=2, verbose=3, color_plot="black", shade_col="gray", 
                 cmap="viridis", d_cmap="Set1"):
       
        self.db = db.lower()
        if f_name is not None:
            self.df = readbib.read_bibfile(f_name, self.db)
        elif df is not None:
            self.df = df
        elif bib_file is not None:
            self.df, self.db = bib_file
        else:
            print("No input file given.\n")
            return None     
        if self.db not in ["scopus", "wos"]:
            print("Warning. No database given.\n")
            return None
        self.n = len(self.df)
        
        self.verbose = verbose
        self.save_results = save_results
        self.res_folder = os.getcwd() + "\\" + res_folder
        
        self.title_var_orig = utilsbib.choose_first(["Title", "Titles", "title", "titles", "cleaned title", "cleaned titles"], self.df) 
        self.abstract_var_orig = utilsbib.choose_first(["Abstract", "abstract", "cleaned abstract"], self.df) 
       
        self.title_var, self.abstract_var = self.title_var_orig, self.abstract_var_orig
        self.current_year = datetime.datetime.now().year
        
        
        if drop_irrelevant:
            self.df = miscbib.drop_irrelevant_features(self.df)
            if self.verbose > 2:
                print("Irrelevant features dropped.\n")
       
        if pre_compute >= 3:
            self.missings_df, self.missings = utilsbib.check_missing_values(self.df)
        
        # computation of additional variables and dataframes 
        if pre_compute >= 1: # topic variables - lowercase
            for c in ["Title", "Titles", "Abstract", "Author Keywords", "Index Keywords", "Indexed Keywords"]:
                if c in self.df.columns:
                    self.df[c.lower()] = self.df[c].fillna("").str.lower()
            self.compute_add_features()
            self.get_mappings()
            
            if self.verbose > 2:
                print("Additional features computed: citations per year, doc names, country of corresponding author, country collaboration.\n")
        if pre_compute >= 2: # features: Doc ID, 
            self.df["Doc ID"] = [f"Doc {i}" for i in range(1, self.n+1)]
            if self.verbose > 3:
                print("Doc ID added as a column in df.\n")
            
            self.preprocess_abstracts()
            self.preprocess_titles()
           
        if pre_compute >= 3: # missings, 
            if self.save_results:
                my_output(self, "missings_df", f_name="\\tables\\" + "missings.xlsx")

        if pre_compute >= 4:
            pass
               
        self.title_var = utilsbib.choose_first(["cleaned title", "cleaned titles", "title", "titles", "Title", "Titles"], self.df) 
        self.abstract_var = utilsbib.choose_first(["cleaned abstract", "abstract", "Abstract"], self.df) 
        self.ak_var = utilsbib.choose_first(["cleaned author keywords", "author keywords", "Author Keywords"], self.df) 
        self.ik_var = utilsbib.choose_first(["cleaned index keywords", "cleaned indexed keywords", "index keywords", "indexed keywords", "Index keywords", "Indexed keywords"], self.df) 
        
        sub_folders = ["plots", "tables", "reports", "indicators", "networks"] 
        if self.save_results:
            
            folders = [self.res_folder + "\\" + s for s in sub_folders]
            utilsbib.make_folders(folders)

        self.fd = os.path.dirname(__file__)
        
        if self.db == "scopus":
            try:
                self.df_scopus_codes = pd.read_excel(self.fd + "\\additional files\\scopus subject area codes.xlsx")
                if self.verbose > 2:
                    print("Scopus subject area codes sucessfully loaded.\n")
            except:
                pass
    
        self.images = []    
        self.dpi, self.sig_digits = dpi_plots, sig_digits
        self.color_plot, self.shade_col = color_plot, shade_col
        self.cmap, self.d_cmap = cmap, d_cmap
        
        self.lang_output, self.lang_docs = lang_output, lang_docs
        
        

        self.kw_var = miscbib.which_keywords(self.df, which=default_keywords)

        if self.db == "scopus" and get_sciences:
            try:
                self.scopus_metadata = pd.read_excel(self.fd + "\\additional files\\Scopus_metadata.xlsx")
            except:
                pass
            try:
                self.add_sources_data_scopus()
            except:
                pass

            
    def save_to_file(self, file_name="biblio.pkl", exclude_dataset=True):
        with open(self.res_folder + "\\" + file_name, "wb") as file:
            import pickle
            if exclude_dataset:
                df0 = self.df
                self.df = None
            pickle.dump(self,  file)
            self.set_data(df0)
        print(f"Analysis saved to {file_name}")
        
    def set_data(self, df):
        self.df = df
        
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
        
    def show_df(self, att):
        utilsbib.make_folders([self.res_folder + "\\tmp"])
        f_name = self.res_folder + f"\\tmp\\{att}.xlsx"
        df = getattr(self, att)
        df.to_excel(f_name, index=False)
        os.startfile(f_name)
        
    # preprocessing
    
    def count_occurences(self):
        self.df = miscbib.count_and_store(self.df,
                                          [("Author(s) ID", "; ", "Number of authors"),
                          ("Title", " ", "Number of words in title"),
                          ("Abstract", " ", "Number of words in abstract"),
                          ("Author Keywords", "; ", "Number of author keywords")])
    
    def get_topic_var(self):
        self.df = miscbib.get_topic_var(self.df)
        
    def compute_add_features(self):

        try:
            self.df = self.df.replace({pd.NA: np.nan, pd.NaT: np.nan})
        except Exception as e:
            print(f"Error in replacing NA values: {e}")
        try:
            self.count_occurences()
        except Exception as e:
            print(f"Error in counting occurrences: {e}")
        try:
            self.df = miscbib.get_cits_per_year(self.df, self.current_year)
        except Exception as e:
            print(f"Error in getting citations per year: {e}")
        try:
            self.df = miscbib.get_doc_names(self.df)
        except Exception as e:
            print(f"Error in getting document names: {e}")
        try:
            self.df = miscbib.add_ca_country_df(self.df, self.db)
        except Exception as e:
            print(f"Error in adding CA country data: {e}")
        try:
            self.df, self.list_countries = miscbib.extract_affs_and_countries(self.df, self.db)
        except Exception as e:
            print(f"Error in extracting affiliations and countries: {e}")
        try:
            self.get_country_collaboration_df()
        except Exception as e:
            print(f"Error in getting country collaboration data: {e}")
        try:
            self.get_topic_var()
        except Exception as e:
            print(f"Error in getting topic variables: {e}")
    
            
    def compute_interdisciplinarity(self, measure=entropy, add_counts=True,
                                    add_f_name="sources_data_short.xlsx"):
        if self.db == "scopus":
            df_sources = pd.read_excel(self.fd+"\\additional files\\" + add_f_name)
            df_scopus_codes = pd.read_excel(self.fd+"\\additional files\\scopus subject area codes.xlsx")
            d_codes = df_sources.set_index("Source Title (Medline-sourced journals are indicated in Green)").to_dict()["All Science Journal Classification Codes (ASJC)"]
            d1, d2, d3 = [df_scopus_codes.set_index("code").to_dict()[f"level {l}"] for l in range(1,4)]
            self.df = miscbib.add_interdisciplinarity_scopus_df(self.df, d_codes, d1, measure=measure, add_counts=add_counts)
            print("Interdisciplinarity computed and stored in self.df")
        else:
            print("Not supported yet")

    def get_concepts(self, df_concepts, topic_var="author keywords", concept_prefix="", kind=1):
        # če je kind=1, je df_concepts dataframe, kjer imaš v imenu stolpca ime koncepta, spodaj pa so naštete ključne besede, ki definirajo koncept
        # če je kind=2, je df_concepts dataframe, kjer imaš v prvem stolpcu koncept, v drugem pa besede, ki se nanj navezujejo

        if kind == 1:
            for c in df_concepts.columns:
                words = df_concepts[c].dropna().tolist()
                self.df[concept_prefix+c] = self.df[topic_var].str.contains("|".join(words)).fillna(0).astype(int)
            self.concept_vars = [concept_prefix+c for c in df_concepts.columns]
        elif kind == 2:
            for c in df_concepts[df_concepts.columns[0]].unique():
                words = df_concepts[df_concepts[df_concepts.columns[0]]==c][df_concepts.columns[1]].tolist()
                self.df[concept_prefix+c] = self.df[topic_var].str.contains("|".join(words)).fillna(0).astype(int)
            self.concept_vars = [concept_prefix+c for c in df_concepts[df_concepts.columns[0]].unique()]
        self.concepts_df = self.df[self.concept_vars]

    
        
    def lower_topic_vars(self, keep_orig=False):
        for v in ["Abstract", "Title", "Titles", "Author Keywords", "Index Keywords", "Indexed Keywords"]:
            if v in self.df.columns:
                self.df[v.lower()] = self.df[v].astype(str).str.lower()
                if keep_orig == False:
                    self.df = self.df.drop(columns=[v])
        if self.verbose > 2:
            print("Topic variables: Abstract, Title, Author Keywords, Index Keywords converted to lowercase.\n")
        
    def get_mappings(self):
        self.dct_sources_abb = None
        try:
            self.dct_sources_abb = miscbib.get_sources_abb_dict(self.df)
        except Exception as e:
            print(f"Error in getting sources abbreviation dictionary: {e}")
        
        try:
            self.dct_au = miscbib.get_dict_of_authors(self.df, db=self.db)
        except Exception as e:
            print(f"Error in getting dictionary of authors: {e}")
        
        try:
            self.dct_au_reverse = {v: k for k, v in self.dct_au.items()}
        except Exception as e:
            print(f"Error in reversing author dictionary: {e}")
        
        try:
            self.dct_sources_issn = miscbib.get_sources_issn_dict(self.df)
        except Exception as e:
            print(f"Error in getting sources ISSN dictionary: {e}")

        
    def preprocess_abstracts(self, exclude=[], remove_numbers=True,
                             replacer_d={}): # , lf=utilsbib.lf, exclude=["nan"], min_len=2, force=0
        
        self.df["cleaned abstract"] = self.df[self.abstract_var].apply(lambda x: utilsbib.text_preprocess(x, exclude=exclude, remove_numbers=remove_numbers))
        self.df["cleaned abstract"] = self.df["cleaned abstract"].map(
            lambda x: utilsbib.preprocess_text(x, replacer_d=replacer_d, remove=exclude))
        
    def preprocess_titles(self, exclude=[], remove_numbers=True,
                          replacer_d={}, remove=[]):
        self.df["cleaned title"] = self.df[self.title_var].apply(lambda x: utilsbib.text_preprocess(x, exclude=exclude, remove_numbers=remove_numbers))
        self.df["cleaned title"] = self.df["cleaned title"].map(
            lambda x: utilsbib.preprocess_text(x, replacer_d=replacer_d, remove=remove))
        
    def preprocess_topic(self, exclude=[], remove_numbers=True,
                             replacer_d={}, remove=[]): # , lf=utilsbib.lf, exclude=["nan"], min_len=2, force=0

        if "topic" in self.df.columns:
            self.df["cleaned topic"] = self.df["topic"].apply(lambda x: utilsbib.text_preprocess(x, exclude=exclude, remove_numbers=remove_numbers))
        if "topic 2" in self.df.columns:
            self.df["cleaned topic 2"] = self.df["topic 2"].apply(lambda x: utilsbib.text_preprocess(x, exclude=exclude, remove_numbers=remove_numbers))
        
    def add_stats_from_refs(self, min_year=1950, max_year=datetime.datetime.now().year,
                        stats=[("Average year from refs", np.mean), 
                               ("Number of references", len),
                               ("Span of years from refs", miscbib.v_range)]):
        self.df, self.ref_years_0 = miscbib.add_stats_from_refs(
            self.df, self.db, min_year=min_year, max_year=max_year, stats=stats)
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
        if self.verbose > 2:
            print(self.new_vars_terms, " computed")
        
    def add_sources_data_scopus(self, f_name="sources_data.xlsx", s_name="Scopus Sources Oct. 2024", **kwds): # tole je delalo s_name="Scopus Sources May 2023"
        if self.db != "scopus":
            print("Works for Scopus db only")
            return None

        print("lalalalala")

        if "Codes" not in self.df.columns:
            try:
                df_sources = pd.read_excel(self.fd+"\\additional files\\" + f_name, sheet_name=s_name)
            except:
                print("Download source info file from Scopus webpage.\n")
                return None
            self.df = pd.concat([self.df, miscbib.add_source_info(self.df, df_sources, **kwds)], axis=1)
        self.df = miscbib.translate_scopus_codes(self.df)
        
        print("lelele")
        
    # various computed features
    
    def compute_misc_features(self):
        if "Title" in self.df.columns or "title" in self.df.columns:
            t_var = "title" if "title" in self.df.columns else "Title"
            self.df["Number of words in title"] = self.df[t_var].map(utilsbib.count_words_string)
            punc = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
            cn = lambda l: sum([1 for x in l if x in punc ])
            if "Title" in self.df.columns: # title (with small t) can be without punctuations
                self.df["Number of punctuations in title"] = self.df["Title"].map(cn)
        if "Abstract" in self.df.columns or "abstract" in self.df.columns or "abstract lemmatized" in self.df.columns:
            if "abstract lemmatized" in self.df.columns:
                a_var = "abstract lemmatized"
            else:
                a_var = "abstract" if "abstract" in self.df.columns else "Abstract"
            self.df["Number of words in abstract"] = self.df[a_var].map(utilsbib.count_words_string)
        self.df = miscbib.count_self_ref_sources(self.df)
        if self.verbose > 2:
            print("Some other feaures computed: lemmatized abstract, number of words in abstract.\n")
    
    def add_periods(self, cut_points=None, num_periods=None, right=False): # adds period column 
        self.df = miscbib.add_period_column(self.df, cut_points=cut_points, num_periods=num_periods, right=right)
    
    # filtering
    
    def filter_documents(self, **kwds):
        # keyword arguments: year_range=None, min_citations=-1, include_types=[], exclude_types=[], include_keywords=[], exclude_keywords=[], include_in_topic=[], exclude_from_topic=[], topic_var="abstract", languages=[], include_disciplines=[], exclude_disciplines=[], include_sources=[], exclude_sources=[], discipline_var="Scopus Sub-Subject Area"
        n = len(self.df)
        self.df = miscbib.filter_documents(self.df, self.db, verbose=self.verbose, **kwds)
        if self.verbose > 1:
            if len(self.df) < n:
                print(f"Dataset size decreased from {n} to {len(self.df)}.\n")
        
    # decsriptives
        
    def get_main_info(self, exclude=[], collab_index=True, spectroscopy=True, min_year=1950, exclude_last_year_growth=True, top5=True, f_name="main info"):
        self.gl_perf_dict, self.ref_stats, self.ak_stats, self.sources_stats, self.lang_stats, self.doc_type_stats, self.other_stats, self.perc_growths, self.top5 = {}, {}, {}, {}, {}, {}, {}, {}, {}
        vrs = [c for c in self.df.columns if c not in exclude]
        if "Year" in vrs:
            self.year_stats = miscbib.year_stats_df(self.df)
            try:
                self.get_production(exclude_last=False)
                self.perc_growths = miscbib.percentage_growth(self.production_df, exclude_last=exclude_last_year_growth)
            except:
                pass
        if "Source title" in vrs:
            if not hasattr(self, "n_sources"):
                self.n_sources = len(self.df["Source title"].unique())
                self.sources_stats["Number of sources"] = self.n_sources
                self.sources_stats["Number of documents per sources"] = self.n / self.n_sources
        if "Language of Original Document" in self.df.columns:
            self.lang_counts = self.df["Language of Original Document"].value_counts()
            
            self.lang_stats["Most frequent language"] = self.lang_counts.index[0]
            self.lang_stats["Relative frequency (of most freqeunt language)"] = self.lang_counts[0] / self.n
            
            self.lang_stats["Number of multilanguage documents"] = self.lang_counts[self.lang_counts.index.str.contains("; ")].sum()
        if "Document Type" in self.df.columns:
            self.doc_type_counts = self.df["Document Type"].value_counts()
            
            self.doc_type_stats["Most frequent type of document"] = self.doc_type_counts.index[0]
            self.doc_type_stats["Relative frequency (of most freqeunt document type)"] = self.doc_type_counts[0] / self.n
        if "Cited by" in vrs:
            self.cits_stats = miscbib.cits_stats_df(self.df)
        if "References" in vrs:
            self.ref_stats = miscbib.occ_stats_df(self.df, "References", sep="; ")
            if spectroscopy:
                if not hasattr(self, "ref_years"):
                    self.add_stats_from_refs(min_year=min_year)
                self.ref_stats["Average year of reference"] = np.mean(self.ref_years)
                self.ref_stats["Correlation (publication year-avg reference year)"] = self.df[["Year", "Average year from refs"]].corr()["Year"]["Average year from refs"]
        if "Author Keywords" in vrs:
            self.ak_stats = miscbib.occ_stats_df(self.df, "Author Keywords", sep="; ")
        if "Index Keywords" in vrs:
            self.ik_stats = miscbib.occ_stats_df(self.df, "Index Keywords", sep="; ")
        # poglej, kje bi dal [No author id available] kot dodatek za rm
        rm = ["", " ", "[No author id available]"]
        if "Author(s) ID" in vrs:
            self.au_stats = miscbib.occ_stats_df(self.df, "Author(s) ID", sep="; ") # tule spremeni, da doaš rm
        elif "Authors" in vrs:
            self.au_stats = miscbib.occ_stats_df(self.df, "Authors", sep=", ")
        if collab_index:
            if "Author(s) ID" in vrs:
                self.au_stats.update(miscbib.collab_stats(self.df["Author(s) ID"], "; ", rm=rm))
            elif "Authors" in vrs:
                self.au_stats.update(miscbib.collab_stats(self.df["Authors"], "; ", rm=rm))
        if "Open Access" in vrs:
            self.other_stats["Number of documents in open access"] = self.df["Open Access"].value_counts().sum()
            self.other_stats["Proportion of documents in open access"] = self.df["Open Access"].value_counts().sum() / self.n
        if ("Abstract" in vrs) or ("abstract" in vrs):
            a_var = "abstract" if "abstract" in vrs else "Abstract"
            self.other_stats["Average number of characters in abstract"] = self.df[a_var].str.len().mean()
            self.other_stats["Average number of words in abstract"] = self.df[a_var].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        if ("Title" in vrs) or ("title" in vrs):
            t_var = "title" if "title" in vrs else "Title"
            self.other_stats["Average number of characters in title"] = self.df[t_var].str.len().mean()
            self.other_stats["Average number of words in title"] = self.df[t_var].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        if "Page count" in vrs:
            self.other_stats["Average page count"] = self.df["Page count"].mean()
            self.other_stats["Length of longest document (in pages)"] = self.df["Page count"].max()
       
        
        self.gl_perf_dict = dict(miscbib.get_perf_ind(self.df, level=3))
        if "H-index" in self.gl_perf_dict:
            self.df_h_score = self.df.sort_values(
                "Cited by", ascending=False).head(self.gl_perf_dict["H-index"])
        # dodaj statistiko o državah
        if hasattr(self, "country_collab_df"):
            if len(self.country_collab_df) > 0:
                self.int_collab_index = self.country_collab_df["MCP"].sum() / self.country_collab_df["Number of documents"].sum()
                self.int_collab_index = np.round(self.int_collab_index, self.sig_digits)
                self.au_stats.update({"International collboration index": self.int_collab_index})
                self.au_stats.update({"Number of countries (by corresponding author)": len(self.country_collab_df)})
        
        if top5:
            try:
                self.count_sources()
                self.top5.update({"Top 5 sources": ",\n".join(self.sources_df["Source title"][:5])})
            except:
                pass
            try:
                self.count_authors()
                self.top5.update({"Top 5 authors": ",\n".join(self.authors_df["Author"][:5])})
            except:
                pass
            try:
                self.count_ca_countries()
                self.top5.update({"Top 5 CA countries": ",\n".join(self.ca_countries_df["CA Country"][:5])})
            except:
                pass
            try:
                self.count_keywords()
                self.top5.update({"Top 5 keywords": ",\n".join(self.keywords_df["Keyword"][:5])})
            except:
                pass      
        
        self.main_info_df = utilsbib.df_from_list_of_dcts([{"number of documents": self.n}, self.sources_stats, self.year_stats, self.cits_stats, self.perc_growths, self.gl_perf_dict, self.top5, self.doc_type_stats, self.au_stats, self.ref_stats, self.ak_stats, self.lang_stats, self.other_stats])
        self.main_info_df["value"] = self.main_info_df["value"].map(lambda x: utilsbib.smart_round(x, ndig=self.sig_digits))
        self.main_info_df_tex = self.main_info_df.to_latex(index=False)
        
        if f_name is not None:
            my_output(self, "main_info_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def preprocess_keywords(self, which="aik", lf=utilsbib.lf, replacer_d={}, remove=[]): #
        if which in ["ak", "aik"]:
            kw_var = "author keywords" if "author keywords" in self.df.columns else "Author Keywords"
            self.df["author keywords"] = self.df[kw_var].map(
                lambda x: utilsbib.preprocess_keywords(
                    x, sw_l=self.lang_docs, lf=lf, 
                    s_char="; ", replacer_d=replacer_d, remove=remove))
        if which in ["ik", "aik"]:
            kw_var = "indexed keywords" if "indexed keywords" in self.df.columns else "Indexed Keywords"
            self.df["indexed keywords"] = self.df[kw_var].map(
                lambda x: utilsbib.preprocess_keywords(
                    x, sw_l=self.lang_docs, lf=lf, 
                    s_char="; ", replacer_d=replacer_d, remove=remove))
        if self.verbose > 2:
            print("Keywords preprocessed: lemmatized, replaced with synonims and removed irrelevant ones.\n")
        # sestavi author in index keywords ter keywords
            
    def get_top_cited_docs(self, top=10, where="global", limit_to=None, 
                           add_vars=[], rm_vars=[], f_name="top cited documents"):
        if (where == "global") or (where == "both"):
            self.top_gl_cit_docs_df = miscbib.get_top_gl_cit_docs(
                self.df, top=top, limit_to=limit_to, add_vars=add_vars, rm_vars=rm_vars)
            self.top_gl_cit_docs_df_tex = self.top_gl_cit_docs_df.to_latex(index=False, multirow=True)
            my_output(self, "top_gl_cit_docs_df", f_name="\\tables\\" + f_name + " global.xlsx")
        if where == "local" or "both":
            print("TO DO: local top cited documents.\n")
        
            
    def get_top_cited_references(self, top=10, f_name="top cited references"):
        if not hasattr(self, "refs_df"):
            self.count_references(extended=True)
        self.top_cited_refs_df = self.refs_df.head(top)
        my_output(self, "top_cited_refs_df", f_name="\\tables\\" + f_name + ".xlsx")

    def get_production(self,  fill_empty=True, rng=None, cut_year=None, exclude_last="smart", f_name="scientific production"):
        self.production_df = miscbib.get_production(
                self.df, fill_empty=fill_empty, rng=rng, cut_year=cut_year, exclude_last=exclude_last)
        if hasattr(self, "scopus_metadata"):
            d_gy = self.scopus_metadata.set_index("YEAR").to_dict()["YEAR COUNTS"]
            d_gy = {k: d_gy[k] for k in d_gy if not pd.isna(k)}
            self.production_df["Global number of documents"] = self.production_df["Year"].map(d_gy)        
        my_output(self, "production_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    # counting
    
    def count_sources(self, f_name="sources counts", add_additional_info=False,
                      freqs2=False, fit_lotka=True, compute_indicators=True, thr_ind=1,
                      f_add_name="sources_data.xlsx", s_name="Scopus Sources May 2023",):
        self.sources_df = utilsbib.freqs_from_col(self.df, "Source title",
                                                  rename_dct=miscbib.freqs_rename_dct)
        self.all_sources = self.sources_df["Source title"].tolist()
        self.n_sources = len(self.all_sources)
        if self.dct_sources_abb is not None:
            self.sources_df["Abbreviated Source Title"] = self.sources_df["Source title"].map(self.dct_sources_abb)
        if freqs2:
            self.sources_f2_df = utilsbib.freqs_of_freqs(
                    self.sources_df, freq_var="Number of documents",
                    name_var="Source title", freq_var_2="Number of sources", split_s="; ")
            if fit_lotka:
                self.sources_lotka = utilsbib.compute_lotka_law(self.sources_f2_df["Number of sources"], self.sources_f2_df["Number of documents"],
                                                                verbose=self.verbose)
                
        if compute_indicators:
            items = self.sources_df[self.sources_df["Number of documents"] >= thr_ind]["Source title"].tolist()
            self.sources_df_ind = utilsbib.get_indicator_df(self.df, "Source title", items=items)
            
        if add_additional_info:
            if self.db == "scopus":
                try:
                    self.sources_df["ISSN"] = self.sources_df["Source title"].map(self.dct_sources_issn)
                    df_sources = pd.read_excel(self.fd+"\\additional files\\" + f_add_name, sheet_name=s_name)
                    df_sources = miscbib.add_source_info(self.sources_df, df_sources)
                    self.sources_df = pd.concat([self.sources_df, df_sources], axis=1)
                    self.sources_df = miscbib.translate_scopus_codes(self.sources_df)
                except:
                    print("No additional source info given")
            
            
        my_output(self, "sources_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_sources_stats()")
  
    def count_doc_types(self, f_name="document types counts", compute_indicators=True, thr_ind=1, freqs2=False):
        self.doc_type_df = utilsbib.freqs_from_col(self.df, "Document Type",
                                                  rename_dct=miscbib.freqs_rename_dct)
        self.doc_type_df_tex = self.doc_type_df.to_latex(index=False)
        
        if compute_indicators:
            items = self.doc_type_df[self.doc_types_df["Number of documents"] >= thr_ind]["Document Type"].tolist()
            self.doc_type_df_ind = utilsbib.get_indicator_df(self.df, "Document Type", items=items)
        
        if freqs2:
            self.doc_type_f2_df = utilsbib.freqs_of_freqs(
                    self.doc_type_df, freq_var="Number of documents",
                    name_var="Document Type", freq_var_2="Number of document types", split_s="; ")
        my_output(self, "doc_type_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def count_ca_countries(self, f_name="corresponding author country counts", compute_indicators=True, thr_ind=1, freqs2=False):
        if "CA Country" not in self.df.columns:
            if "Correspondence Address" in self.df.columns:
                self.df = miscbib.add_ca_country_df(self.df, self.db)
        self.ca_countries_df = utilsbib.freqs_from_col(self.df, "CA Country",
                                                       rename_dct=miscbib.freqs_rename_dct)
        
        if compute_indicators:
            items = self.ca_countries_df[self.ca_countries_df["Number of documents"] >= thr_ind]["CA Country"].tolist()
            self.ca_countries_df_ind = utilsbib.get_indicator_df(self.df, "CA Country", items=items)
        
        if freqs2:
            self.ca_countries_f2_df = utilsbib.freqs_of_freqs(
                    self.ca_countries_df, freq_var="Number of documents",
                    name_var="CA Country", freq_var_2="Number of countries", split_s="; ")
        my_output(self, "ca_countries_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_ca_countries_stats()")
        
    def count_authors(self, rm=["", " ", "[No author id available]", "nan"], compute_indicators=True, thr_ind=1, f_name="author counts", freqs2=False):
        if "Author(s) ID" in self.df.columns:
            self.authors_df = miscbib.both_counts_occurrences(self.df["Author(s) ID"], ";", rm=rm)
            self.au_var = "Author(s) ID"
        elif "Authors" in self.df.columns:
            self.authors_df = miscbib.both_counts_occurrences(self.df["Authors"], ", ", rm=rm)
            self.au_var = "Authors"
        self.authors_df["Author"] = self.authors_df["item"].map(self.dct_au)
        self.authors_df = self.authors_df.rename(columns={"item": self.au_var, "count": "Number of documents", "fractional count": "Fractional count of documents", "proportion": "Proportion of documents", "fractional proportion": "Fractional proportion of documents"})
        no = ["Author"] + self.authors_df.columns[:-1].tolist()
        self.authors_df = self.authors_df[no]
        
        if compute_indicators:
            items = self.authors_df[self.authors_df["Number of documents"] >= thr_ind]["Author"].tolist()
            self.authors_df_ind = utilsbib.get_indicator_df(self.df, "Author", items=items)
        
        if freqs2:
            self.authors_f2_df = utilsbib.freqs_of_freqs(
                    self.authors_df, freq_var="Number of documents",
                    name_var="Author", freq_var_2="Number of authors", split_s="; ")
        my_output(self, "authors_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_authors_stats()")
        
    def count_areas(self, f_name="", compute_indicators=True, thr_ind=1, summary=True, k_fields=10):
        if self.db == "scopus":
            if "Codes" not in self.df.columns:
                print("Codes not available.\n")
                return None
            else:
                d0 = {"count": "Number of documents", "fractional count": "Fractional count of documents", "proportion": "Proportion of documents", "fractional proportion": "Fractional proportion of documents"}
                
                def un(x):
                    if x == x:
                        return "; ".join(list(set(x.split("; "))))
                    return x
                
                self.fields_df = miscbib.both_counts_occurrences(self.df["Fields"].map(un), "; ")
                d0.update({"item": "Field"})
                self.fields_df = self.fields_df.rename(columns=d0)
                self.areas_df = miscbib.both_counts_occurrences(self.df["Areas"].map(un), "; ")
                d0.update({"item": "Area"})
                self.areas_df = self.areas_df.rename(columns=d0)
                self.sciences_df = miscbib.both_counts_occurrences(self.df["Sciences"].map(un), "; ")
                d0.update({"item": "Science"})
                self.sciences_df = self.sciences_df.rename(columns=d0)
                try:
                    df_scopus_codes = pd.read_excel(self.fd + "\\additional files\\scopus subject area codes.xlsx")
                    d_fs = df_scopus_codes.set_index("level 3").to_dict()["level 1"]
                    d_fa = df_scopus_codes.set_index("level 3").to_dict()["level 2"]
                    d_as = df_scopus_codes.set_index("level 2").to_dict()["level 1"]
                    self.fields_df["Area"] = self.fields_df["Field"].map(d_fa)
                    self.fields_df["Science"] = self.fields_df["Field"].map(d_fs)
                    self.areas_df["Science"] = self.areas_df["Area"].map(d_as)
                except:
                    pass               
                
                try:
                    if compute_indicators:
                        items = self.fields_df[self.fields_df["Number of documents"] >= thr_ind]["Field"].tolist()
                        self.fields_df_ind = utilsbib.get_indicator_df(self.df, "Field", items=items, search="substring")
                        items = self.areas_df[self.areas_df["Number of documents"] >= thr_ind]["Area"].tolist()
                        self.areas_df_ind = utilsbib.get_indicator_df(self.df, "Area", items=items, search="substring")
                        items = self.sciences_df[self.sciences_df["Number of documents"] >= thr_ind]["Science"].tolist()
                        self.sciences_df_ind = utilsbib.get_indicator_df(self.df, "Science", items=items, search="substring")
                except:
                    pass
                
                my_output(self, "fields_df", f_name="\\tables\\fields.xlsx")
                my_output(self, "areas_df", f_name="\\tables\\areas.xlsx")
                my_output(self, "sciences_df", f_name="\\tables\\sciences.xlsx")
                
                if summary:
                    df0, df1, df2 = self.fields_df, self.areas_df, self.sciences_df
                    
                    l1, l2 = [], []                  
                    dfs = pd.DataFrame()
                    
                    dfs["Science"] = df2.apply(lambda row: f"{row['Science']} ({round(row['Proportion of documents']*100, 2)}%)", axis=1)
                    
                    for v in df2["Science"]:
                        df1_ = df1[df1["Science"]==v]
                        df1_["Conditional proportion"] = df1_["Number of documents"] / df1_["Number of documents"].sum()
                        ll1 = "\n".join(df1_.apply(lambda row: f"{row['Area']} ({round(row['Conditional proportion']*100, 2)}%)", axis=1)) 
                        df0_ = df0[df0["Science"]==v]
                        df0_ = df0_.sort_values("Number of documents", ascending=False)
                        df0_["Conditional proportion"] = df0_["Number of documents"] / df0_["Number of documents"].sum()
                        if k_fields is not None:
                            df0_ = df0_.head(k_fields)
                        ll2 = "\n".join(df0_.apply(lambda row: f"{row['Field']} ({round(row['Conditional proportion']*100, 2)}%)", axis=1))    
                        l1.append(ll1)
                        l2.append(ll2)
                        
                    dfs["Areas (all)"] = l1
                    if k_fields is not None:
                        dfs[f"Fields (top {k_fields})"] = l2
                    else:
                        dfs["Fileds (all)"] = l2
                    self.summary_fields_df = dfs
                    my_output(self, "summary_fields_df", f_name="\\tables\\summary fields.xlsx")

        else:
            print("Not supported")
        
    def count_keywords(self, which="ak", compute_indicators=True, thr_ind=5, f_name="keyword counts", freqs2=False):
        if which == "ak":
            kw = [c for c in self.df.columns if c in ["author keywords", "Author Keywords"]][0]
            self.df["keywords"] = self.df[kw]
        elif which == "ik":
            kw = [c for c in self.df.columns if c in ["indexed keywords", "Indexed Keywords"]][0]
            self.df["keywords"] = self.df[kw]
        self.kw_var = miscbib.which_keywords(self.df, which=which)
        self.keywords_df = miscbib.count_occurrences_df(self.df[self.kw_var], "; ")
        self.keywords_df = self.keywords_df.rename(columns={"item": "Keyword"})
        
        if compute_indicators:
            items = self.keywords_df[self.keywords_df["Number of documents"] >= thr_ind]["Keyword"].tolist()
            self.keywords_df_ind = utilsbib.get_indicator_df(self.df, kw, items=items, search="substring")
        
        if freqs2:
            self.keywords_f2_df = utilsbib.freqs_of_freqs(
                    self.keywords_df, freq_var="Number of documents",
                    name_var="Keyword", freq_var_2="Number of document keywords", split_s="; ")
        my_output(self, "keywords_df", f_name="\\tables\\" + f_name + " " + which + ".xlsx", additional="get_keywords_stats()")
        
    def count_references(self, extended=False, compute_indicators=False, top=None, f_name="references counts"):
        self.refs_df = miscbib.count_occurrences_df(self.df["References"], "; ")
        self.refs_df = self.refs_df.rename(columns={"item": "Reference"})

        if extended:
            self.cited_auths_list = [miscbib.extract_info_refs_scopus(r)[0] for r in self.refs_df["Reference"]]
            r_df = pd.DataFrame([miscbib.extract_info_refs_scopus(r)[1:] for r in self.refs_df["Reference"]], columns=["First Author", "Authors", "Year", "Title", "Source title"])
            self.refs_df = pd.concat([self.refs_df, r_df], axis=1)
            self.refs_df["Source title"] = self.refs_df["Source title"].astype(str)
            #self.refs_df["First Author"] = self.refs_df["Authors"].map(lambda x: str(x).split("., ")[0])
            self.refs_df["Short name"] = self.refs_df["First Author"] + " (" + self.refs_df["Year"].astype(str) + ")"
            try:
                self.refs_df["Short name"] = self.refs_df["Short name"].map(lambda x: x.replace(".0",""))
            except:
                pass
        self.refs_df = self.refs_df.rename(columns={"Number of documents": "Number of local citations",
                                                    "Proportion of documents": "Proportion of local citations"})
        self.refs_df = self.refs_df[self.refs_df["Reference"].map(lambda x: utilsbib.check_balance(x))]
        self.refs_df = self.refs_df.sort_values("Number of local citations", ascending=False)
        
        self.refs_df = self.refs_df[self.refs_df["Reference"].str.len() > 10] # removes references that contain just year
        
        my_output(self, "refs_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_refs_stats()")
        if compute_indicators:
            top = len(self.refs_df) if top is None else top
            self.refs_df_ind = utilsbib.get_indicator_df(
                self.df, "References", search="substring", items=self.refs_df["Reference"].tolist()[:top])
            
    def count_words_and_phrases(self, text_column="cleaned abstract", compute_indicators=False, top=10, max_n=2, min_f=0.01): #min_f še ne dela
        if text_column not in self.df.columns:
            print("Wrong text column")
            return None
        
        df_counts = utilsbib.count_ngrams(self.df, text_column, max_n)
        
        if text_column in ["abstract", "cleaned abstract", "Abstract"]:
            self.abs_words_df = df_counts
            if compute_indicators:
                self.abs_words_df_ind = utilsbib.get_indicator_df(
                    self.df, text_column, search="substring", items=df_counts["word/phrase"].tolist()[:top])
        elif text_column in ["title", "titles", "Title", "Titles", "cleaned title"]:
            self.tit_words_df = df_counts
            if compute_indicators:
                self.tit_words_df_ind = utilsbib.get_indicator_df(
                    self.df, text_column, search="substring", items=df_counts["word/phrase"].tolist()[:top])
        elif text_column in ["topic", "topic2"]:
            self.topic_words_df = df_counts
            if compute_indicators:
                self.topic_words_df_ind = utilsbib.get_indicator_df(
                    self.df, text_column, search="substring", items=df_counts["word/phrase"].tolist()[:top])
                
        
    def count_local_cited_authors(self, f_name="local cited authors counts"):
        if not hasattr(self, "refs_df"):
            self.count_references(extended=True)
        lal = [self.cited_auths_list[i] * self.refs_df["Number of local citations"].iloc[i] for i in range(len(self.refs_df))]
        la = utilsbib.unnest(lal)
        self.local_cited_authors = utilsbib.freqs_from_list(la, order=2)
        self.local_cited_authors_df = pd.DataFrame(self.local_cited_authors, columns=["Author", "Number of local citations"])
        self.local_cited_authors_df = self.local_cited_authors_df[self.local_cited_authors_df["Author"] != "Et al"]
        my_output(self, "local_cited_authors_df", f_name="\\tables\\" + f_name + ".xlsx", additional="get_words_and_phrases_stats()")
    
    def count_local_cited_sources(self, f_name="local cited sources coutns"):
        if not hasattr(self, "refs_df"):
            self.count_references(extended=True)
        self.local_cited_sources_df = self.refs_df.groupby("Source title").agg(sum).reset_index().sort_values("Number of local citations", ascending=False)
        self.local_cited_sources_df = self.local_cited_sources_df.drop(columns=["Proportion of local citations", "Year"])
        self.local_cited_sources_df = self.local_cited_sources_df[self.local_cited_sources_df["Source title"]!=""]
        my_output(self, "local_cited_sources_df", f_name="\\tables\\" + f_name + ".xlsx")
    
    def get_all_sources_indicators(self):
        if not hasattr(self, "sources_df"):
            self.count_sources()
        items = self.sources_df["Source title"].tolist()
        self.sources_all_df_ind = utilsbib.get_indicator_df(self.df, "Source title", items=items)
        
    def get_all_fields_indicators(self):
        if not hasattr(self, "fields_df"):
            self.count_areas()
        items = self.fields_df["Field"].tolist()
        self.fields_all_df_ind = utilsbib.get_indicator_df(self.df, "Fields", items=items, search="substring")
    
    # za ostale: ključne besede, reference, avtorje ... bi bilo tega preveč - morda dodati samo še za države
    # že filedov je lahko zelo veliko
    
    
    def get_sources_stats(self, items=[], exclude=["nan"], 
                          top=20, top_by="Number of documents", 
                          min_freq=2, cov_prop=0.8, level=1, compute_indicators=True,
                          abbreviate=True, f_name="top sources stats"):
        if not hasattr(self, "sources_df"):
            self.count_sources()
        self.sources_df = miscbib.get_add_stats(
            self.df, self.sources_df, "Source title", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop, level=level)
        items = miscbib.select_items(
            self.sources_df, "Source title", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, freq_col="Number of documents", cov_prop=cov_prop)
        if compute_indicators:
            self.sources_df_ind = utilsbib.get_indicator_df(self.df, "Source title", items=items)
            if abbreviate:
                self.sources_df_ind = self.sources_df_ind.rename(columns=self.dct_sources_abb)
        if f_name is not None:
            my_output(self, "sources_df", f_name="\\tables\\" + f_name + ".xlsx")
        

        
    def get_ca_countries_stats(self, items=[], exclude=["nan"], 
                               top=20, top_by="Number of documents", 
                               min_freq=5, cov_prop=0.8, level=1, compute_indicators=True,
                               f_name="top ca countries stats"):
        # ideja: če daš items="eu" ali pa items="europe", ti izbere samo ustrezne države
        if type(items) == str:
            if items.lower() == "eu":
                items = miscbib.eu_countries
        if not hasattr(self, "ca_countries_df"):
            self.count_ca_countries()
        self.ca_countries_df = miscbib.get_add_stats(
            self.df, self.ca_countries_df, "CA Country", 
            items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, cov_prop=cov_prop, level=level)
        items = miscbib.select_items(
            self.ca_countries_df, "CA Country", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, freq_col="Number of documents", cov_prop=cov_prop) # morda za eu ne dela ta del
        if compute_indicators:
            self.ca_countries_df_ind = utilsbib.get_indicator_df(self.df, "CA Country", items=items)
        if f_name is not None:
            my_output(self, "ca_countries_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def get_authors_stats(self, items=[], exclude=["nan", "", " ", "[No author id available]"], 
                          top=20, top_by="Number of documents", 
                          min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True, f_name="top authors stats"):                 
        if not hasattr(self, "authors_df"):
            self.count_authors()
        items = miscbib.select_items(
            self.authors_df, "Author", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq,
            freq_col=freq_col, cov_prop=cov_prop)
        items = [self.dct_au_reverse[it] for it in items if it==it]
        ps = miscbib.get_performances(self.df, "Author(s) ID", items, search="substring", level=level)
        self.authors_df = pd.merge(self.authors_df, ps, on="Author(s) ID")
        self.authors_df = miscbib.order_by_cits(self.authors_df)
        if compute_indicators:
            self.authors_df_ind = utilsbib.get_indicator_df(
                self.df, "Author(s) ID", search="substring", items=items)
            self.authors_df_ind = self.authors_df_ind.rename(columns=self.dct_au) 
        if f_name is not None:
            my_output(self, "authors_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def get_keywords_stats(self, which="ak", items=[], exclude=["nan"], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True, f_name="top keywords stats"): 
        if not hasattr(self, "keywords_df"):
            self.count_keywords(which=which)
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
        if f_name is not None:
            my_output(self, "keywords_df", f_name="\\tables\\" + f_name + ".xlsx")
            
    def get_fields_stats(self, items=[], exclude=["nan"], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True, f_name="top fields stats"):                 
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
        my_output(self, "fields_df", f_name="\\tables\\" + f_name + ".xlsx")
        
        
    def get_areas_stats(self, items=[], exclude=["nan"], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True, f_name="top areas stats"):                 
        if not hasattr(self, "areas_df"):
            self.count_areas()
        #if type(items) == str:
        #    items = [k for k in self.keywords_df["Keyword"] if items in k]
        items = miscbib.select_items(
            self.areas_df, "Area", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, 
            freq_col=freq_col, cov_prop=cov_prop)
        ps = miscbib.get_performances(self.df, "Areas", items, search="substring", level=level)
        ps = ps.rename(columns={"Areas": "Area"})
        self.areas_df = pd.merge(self.areas_df, ps, on="Area")
        self.areas_df = miscbib.order_by_cits(self.areas_df)
        if compute_indicators:
            self.areas_df_ind = utilsbib.get_indicator_df(
                self.df, "Areas", search="substring", items=items)
        my_output(self, "areas_df", f_name="\\tables\\" + f_name + ".xlsx")
        
        
    def get_sciences_stats(self, items=[], exclude=["nan"], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="Number of documents", level=1,
                          compute_indicators=True, f_name="top areas stats"):                 
        if not hasattr(self, "sciences_df"):
            self.count_areas()
        #if type(items) == str:
        #    items = [k for k in self.keywords_df["Keyword"] if items in k]
        items = miscbib.select_items(
            self.sciences_df, "Science", items=items, exclude=exclude, 
            top=top, top_by=top_by, min_freq=min_freq, 
            freq_col=freq_col, cov_prop=cov_prop)
        ps = miscbib.get_performances(self.df, "Sciences", items, search="substring", level=level)
        ps = ps.rename(columns={"Sciences": "Science"})
        self.sciences_df = pd.merge(self.sciences_df, ps, on="Science")
        self.sciences_df = miscbib.order_by_cits(self.sciences_df)
        if compute_indicators:
            self.sciences_df_ind = utilsbib.get_indicator_df(
                self.df, "Sciences", search="substring", items=items)
        my_output(self, "sciences_df", f_name="\\tables\\" + f_name + ".xlsx")
    
    def get_words_and_phrases_stats(self, max_n=2, items=[], exclude=["nan"], 
                            top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                          freq_col="total count", level=1, f_name="top words and phrases stats",
                          compute_indicators=True):
        
        for counts_df_ in ["abs_words_df", "tit_words_df", "topic_words_df"]:
            if not hasattr(self, counts_df_):
                continue
            else:
                c_var = {"abs_words_df": "cleaned abstract", "tit_words_df": "cleaned title"}[counts_df_]
            
            counts_df = getattr(self, counts_df_)
            items = miscbib.select_items(
                counts_df, "word/phrase", items=items, exclude=exclude, 
                top=top, top_by="total count", min_freq=min_freq, 
                freq_col=freq_col, cov_prop=cov_prop)
            ps = miscbib.get_performances(self.df, c_var, items, search="substring", level=level)
            ps = ps.rename(columns={c_var: "word/phrase"})
            counts_df = pd.merge(counts_df, ps, on="word/phrase")
            counts_df = miscbib.order_by_cits(counts_df)
            setattr(self, counts_df_, counts_df)
            my_output(self, counts_df_, f_name="\\tables\\" + f_name + ".xlsx") # tukaj usrezno uredi
            if compute_indicators:
                count_df, presence_df, tfidf_df = utilsbib.get_indicator_dfs_text(self.df, c_var, words=counts_df["word/phrase"].tolist())
                setattr(self, counts_df_ + "_ind_c_df", count_df)
                setattr(self, counts_df_ + "_ind_df", presence_df)
                setattr(self, counts_df_ + "_ind_tfidf", tfidf_df)
           
    def get_refs_stats(self, items=[], exclude=["nan"], 
                       top=20, top_by="Number of documents",
                           min_freq=5, cov_prop=0.8, 
                           freq_col="Number of local citations", level=1,
                           compute_indicators=True, f_name="top refs stats"):                  
        # precej nerelevantna zadeva
        if not hasattr(self, "refs_df"):
            self.count_references()
        items = miscbib.select_items(
            self.refs_df, "Reference", items=items, exclude=exclude, 
            top=top, top_by=freq_col, min_freq=min_freq, 
            freq_col=freq_col, cov_prop=cov_prop)
        ps = miscbib.get_performances(self.df, "References", items, search="substring", level=level)
        ps = ps.rename(columns={"References": "Reference"})
        self.refs_df = pd.merge(self.refs_df, ps, on="Reference")
        #self.refs_df = miscbib.order_by_cits(self.refs_df)
        if compute_indicators:
            self.refs_df_ind = utilsbib.get_indicator_df(
                self.df, "References", search="substring", items=items)
        my_output(self, "refs_df", f_name="\\tables\\" + f_name + ".xlsx")
    
    # dynamics
    
    def get_sources_dynamics(self, items=None, max_items=20, fill_empty=True, f_name="source dynamics"):
        self.ind_df_sources = utilsbib.get_indicator_df(self.df, "Source title", items=items, max_items=max_items)
        self.ind_df_sources = self.ind_df_sources.rename(columns=self.dct_sources_abb)
        self.sources_dyn_df = miscbib.get_productions_ind(
            self.ind_df_sources, self.df["Year"], fill_empty=fill_empty)
        my_output(self, "sources_dyn_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def get_ca_country_dynamics(self, items=None, max_items=20, fill_empty=True, f_name="country dynamics"):
        if items == "eu": # še ne dela OK
            items = miscbib.eu_countries
            max_items = 28
        self.ind_df_ca_countries = utilsbib.get_indicator_df(self.df, "CA Country", items=items, max_items=max_items)
        self.ca_country_dyn_df = miscbib.get_productions_ind(
            self.ind_df_ca_countries, self.df["Year"], fill_empty=fill_empty)
        my_output(self, "ca_country_dyn_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    def get_country_collaboration_df(self, f_name="country collaboration"):
        self.country_collab_df = miscbib.get_country_collaboration_df(self.df)
        my_output(self, "country_collab_df", f_name="\\tables\\" + f_name + ".xlsx")
        
    # combinations of different aspects
    
    # relation between two aspects
    
    def relate_two_aspects(self, a1="keywords", a2="sources", df1=None, df2=None, method="cca", nc=2, normalize="jaccard"):
        if a1 is not None:
            df1 = getattr(self, a1+"_df_ind")
            self.rel1 = a1
        if a2 is not None:
            df2 = getattr(self, a2+"_df_ind")
            self.rel2 = a2
        if (df1 is None) or (df2 is None):
            print("No data.\n")
            return None
        self.two_aspects = utilsbib.TwoConcepts(df1, df2)
        self.cross_concepts_df = df1.T.dot(df2)
        if normalize == "jaccard":
            self.cross_concepts_df_jac = pd.DataFrame(utilsbib.normalize_matrix(self.cross_concepts_df), 
                                                      index=self.cross_concepts_df.index, columns=self.cross_concepts_df.columns)
        self.decomposition_method = method
        getattr(self.two_aspects, method)(nc=nc)
    
    
    def mds_df_from_aspects(self, fields=["keywords", "sources", "authors"], 
                            ks=10, customs=[()], nc=2,  metric="cosine",
                            rename_dct_group={}, f_name="mds coordinates"):
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
                print(f"Wrong input for {field}.\n")
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
        my_output(self, "df_mds", f_name="\\tables\\" + f_name + ".xlsx")
        
    #za avtorje ne dela OK
    #def get_authors_dynamics(self, items=None, max_items=20):
    #    self.ind_df_authors = utilsbib.get_indicator_df(self.df, "Authors", items=items, max_items=max_items, search="substring")
        
    # top authors, ca countires ... time production
    
    # topic modelling
    
    def get_topics(self, text_var="Abstract", model=gensim.models.LdaModel,
                   num_topics=None, stop_words=utilsbib.add_stopwords, show_words_per_topic=100,
                   f_name="topics (topic modelling) "):
        if num_topics is None:
            self.get_keyword_co_net(top=100)
            num_topics = self.KW_cooc_net.n_clusters
        
        text_var = text_var.lower() if text_var.lower() in self.df.columns else text_var
        self.lda_model, self.doc_lda, self.id2word, self.corpus \
            = utilsbib.lda_topics(self.df, text_var, model=model, 
                                  num_topics=num_topics, stop_words=stop_words)
        self.top_topics = self.lda_model.top_topics(self.corpus, topn=show_words_per_topic)
        self.topics_df = pd.concat([pd.DataFrame(
            t[0], columns=["w %d " % (i+1) , "topic %d" % (i+1)])
            for i, t in enumerate(self.top_topics)], axis=1)
        my_output(self, "topics_df", f_name="\\tables\\" + f_name + ".xlsx")
      
        
    
    # reporting
         
    def to_excel(self, f_name="report.xlsx", index=0, exclude=[], 
                 autofit=True, M_len=100):
        reportbib.results_to_excel(self, f_name=self.res_folder+"\\reports\\"+f_name, 
                                  index=index, exclude=exclude, 
                                  autofit=autofit, M_len=M_len)
        
    def to_word(self, f_name="report.docx", include_non_core=False):
        reportbib.results_to_word(self, f_name=self.res_folder+"\\reports\\"+f_name,
                                  include_non_core=include_non_core)
        
    def to_latex(self, f_name="report.tex", template_tex="additional files\\report templates\\template report.tex", split_string="TO BE HERE: ("):
        reportbib.results_to_latex(self, f_name=self.res_folder+"\\reports\\"+f_name,
                                   template_tex=self.fd+"\\"+template_tex, split_string=split_string)

    def get_reports(self, formats=["docx", "xlsx", "tex"], f_name="report"):
        for f in formats:
            if f == "docx":
                self.to_word(f_name=f_name+"."+f)
            elif f == "xlsx":
                self.to_excel(f_name=f_name+"."+f)
            elif f == "tex":
                self.to_latex(f_name=f_name+"."+f)

                
        
        
        
