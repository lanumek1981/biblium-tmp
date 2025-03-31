# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:36:23 2022

@author: Lan.Umek
"""

"""
!pip install pyLDAvis -qq
!pip install -qq -U gensim
!pip install spacy -qq
"""

import pandas as pd
import numpy as np
import utilsbib
import os
import datetime
from cdlib import algorithms
import networkx as nx
import re
from functools import reduce
import readbib
from scipy.stats import entropy


def chunks(eids, q_max):
    n = int(len(eids)/q_max)
    return [eids[(i-1)*q_max:i*q_max ] for i in range(1,n+1)] + [eids[n*q_max:]]

def get_query(eid):
    return "".join([" EID("+e+") OR" for e in eid])[1:-3]

def get_queries_from_scopus(fn, q_max=2000, out="queries.csv"):
    df = pd.read_csv(fn, error_bad_lines=False)

    eids = df["EID"].to_list()
    Eid = chunks(eids, q_max)
    Q = [get_query(eid) for eid in Eid]

    df_q = pd.DataFrame(Q, columns=["query"])
    df_q.to_csv(out, index=False)
    return df_q

def get_doi_query_wos_from_scopus(df, out="doi.txt"):
    l = df.dropna(subset=["DOI"])["DOI"].tolist()
    for i in range(1,10):
        l = [ll for ll in l if "(%d" % i not in ll]

    t = " OR ".join(["(DO=(" + ll + "))"for ll in l])
    print(t)
    f = open(out, "wt")
    f.write(t)
    f.close()
    
def one_source_scopus_query(source):
    source = " ".join([s.capitalize() for s in source.split()])
    return "LIMIT-TO ( EXACTSRCTITLE , " + "\" %s\") " % source

    
def query_for_scopus_area(topic="a*", area="Public Administration"):
    df = pd.read_excel(os.path.dirname(__file__)+"\\additional files\\scopus\\citescore.xlsx")
    sources = df[df["Scopus Sub-Subject Area"]==area]["Titles"].unique()
    mq = "OR ".join([one_source_scopus_query(source) for source in sources])
    return "TITLE-ABS-KEY( %s ) AND (" % topic + mq + ")"



def drop_irrelevant_features(df, features=[
        "Issue", "Art. No.", "Page start", "Page end", "Page count", 
        "DOI", "Link", "Molecular Sequence Numbers", "Editors", "Sponsors",
        "Publisher", "Conference name", "Conference date", "Conference location",
        "Conference code", "ISBN", "CODEN", "PubMed ID"], # "ISSN" pustimo zaradi povezovanja z viri
        drop_funding=True):
    dc = [c for c in df.columns if c in features]
    if drop_funding:
        dc += [c for c in df.columns if "Funding" in c]
    return df.drop(columns=dc)


def read_data(f_name=None, db="", df=None):

    
    if f_name is not None: 
        if isinstance(f_name, str):
            return readbib.read_bibfile(f_name, db)
        elif isinstance(f_name, list):
            dfs = []
            for f in f_name:
                df_tmp = readbib.read_bibfile(f, db)
                df_tmp[f"group {f}"] = 1
                dfs.append(df_tmp)
            merged_df = pd.concat(dfs, sort=False)
            
            if db == "scopus":
                ui = "EID"
            else:
                ui = None
                
            if ui is not None:
                d1 = merged_df[[ui] + [c for c in merged_df.columns if "group" in c]]
                d1 = d1.fillna(0)
                d1 = d1.groupby(ui).agg(sum).reset_index()

                d2 = merged_df[[c for c in merged_df.columns if "group" not in c]]
                d2 = d2.drop_duplicates(subset=[ui])

                return pd.merge(d1, d2, on=ui)
            else:
                return merged_df
    elif df is not None:
        return df
    else:
        print("No input file given")
        return None 

# filtering

def filter_documents(df, db, year_range=None, min_citations=-1, 
                         include_types=[], exclude_types=[],
                         include_keywords=[], exclude_keywords=[],
                         include_in_topic=[], exclude_from_topic=[],
                         topic_var="abstract", languages=[],
                         include_disciplines=[], exclude_disciplines=[],
                         include_sources=[], exclude_sources=[],
                         discipline_var="Scopus Sub-Subject Area",
                         bradford=0,
						 verbose=1): # za wos je najbolje discipline_var="research areas"
        # dodati Bradfordov zakon 
        n0 = len(df)  
        if (year_range is not None) and ("Year" in df.columns):
            df = df[df["Year"].between(*year_range)]
        if "Cited by" in df.columns:
            df = df[df["Cited by"] >= min_citations]
        if "Document Type" in df.columns:
            if len(include_types) > 0:
                types = include_types
            else:
                types = df["Document Type"].value_counts().index
                if len(exclude_types) > 0:
                    types = [t for t in types if t not in exclude_types]
            df = df[df["Document Type"].isin(types)]
        if len(include_keywords):
            df = df[df["keywords"].str.contains("|".join(include_keywords)).replace(np.nan, False)]
        if len(exclude_keywords):    
            df = df[~df["keywords"].str.contains("|".join(exclude_keywords)).replace(np.nan, False)]
        if len(include_in_topic):
            df = df[df[topic_var].str.contains("|".join(include_in_topic).lower()).replace(np.nan, False)]        
        if len(exclude_from_topic):
            df = df[~df[topic_var].str.contains("|".join(exclude_from_topic).lower()).replace(np.nan, False)]
        if len(languages):
            df = df[df["Language of Original Document"].str.contains("|".join(languages))]
        if len(include_disciplines):
            if db.lower() == "wos":
                df = df[df[discipline_var].str.contains("|".join(include_disciplines)).replace(np.nan, False)]
            elif db.lower() == "scopus":
                print("Not yet supported")
                #sources = miscbib2.get_sources_scopus_area(include_disciplines, v=discipline_var)
                #df = df[df["Source title"].str.contains("|".join(sources)).replace(np.nan, False)]               
        if len(exclude_disciplines):
            if db.lower() == "wos":
                df = df[~df[discipline_var].str.contains("|".join(exclude_disciplines)).replace(np.nan, True)]
        if len(include_sources):
            df = df[df["Source title"].str.contains("|".join(include_sources)).replace(np.nan, False)]
        if len(exclude_sources):
            df = df[~df["Source title"].str.contains("|".join(exclude_sources)).replace(np.nan, True)]
        if bradford > 0:
            sources_df = utilsbib.freqs_from_col(df, "Source title", rename_dct=freqs_rename_dct)
            sources_df = get_bradford_zones(sources_df, n_zones=bradford)
            bradford_sources = sources_df[sources_df["Zone"] == "Zone 1"]["Source title"].tolist()
            df = df[df["Source title"].str.contains("|".join(bradford_sources)).replace(np.nan, True)]
        
        df = df.reset_index()
        
        if verbose > 1:
            print("Dataset reduced (from %d cases to %d)" % (n0, len(df)))
        return df

# Bradford's law
        
def get_bradford_zones(df, v="Number of documents", n_zones=3):
    df = df.sort_values(v, ascending=False)
    total = df[v].sum()
    df["cumsum %s" % v] = df[v].cumsum() / total
    bins = np.append(np.arange(0, 1, 1/n_zones), 1)
    labels = ["Zone %d" % (i+1) for i in range(n_zones)]
    df["Zone"] = pd.cut(df["cumsum %s" % v], bins=bins, labels=labels)
    df["logrank"] = np.log(df[v].rank(ascending=False))
    return df


# Functions for computation of additional features
    
def add_period_column(df, cut_points=None, num_periods=None, right=False):
    if cut_points is None and num_periods is not None:
        num_periods -= 1
        min_year = df["Year"].min()
        max_year = df["Year"].max()
        cut_points = [min_year + (max_year - min_year) * i / num_periods for i in range(num_periods)]
        cut_points.append(max_year)
    elif not cut_points:
        raise ValueError("Either cut_points or num_periods must be provided.")
    
    # Sort cut points
    cut_points = sorted(cut_points)
    
    # Add only the upper bound, no need for -inf
    cut_points = cut_points + [float("inf")]
    
    # Period labels adjusted to match the exact number of periods needed
    period_labels = [f"Period {i}" for i in range(1, len(cut_points))]

    # Bin data with adjusted cut_points and labels
    df["Period"] = pd.cut(df["Year"], bins=cut_points, labels=period_labels, right=right) 

    return df



def get_cits_per_year(df, cy):
    if "Cited by" in df.columns:
        df["Cited by"] = df["Cited by"].replace({np.nan: 0})
    if "Year" in df.columns: 
        df["Age"] = cy-df["Year"]+1

        df["Age"] = df["Age"].apply(lambda x: x if x > 0 else 1)
        if "Cited by" in df.columns:
            df["Citations per year"] = df["Cited by"] / df["Age"]
    return df


# transformation
    
def transform_group_df(df, name_var="x", group_var="group", vs=["cond_g"]):
    pivot_dfs = []
    
    for v in vs:
        try:
            pivot_df = df.pivot(index=name_var, columns=group_var, values=v).reset_index()
            pivot_df.columns = [name_var] + [f"{v} {col}" for col in pivot_df.columns[1:]]
            pivot_df = pivot_df.fillna(0)
            pivot_dfs.append(pivot_df)
        except:
            pass

    final_df = pivot_dfs[0] if len(pivot_dfs) == 1 else reduce(lambda left, right: pd.merge(left, right, on=name_var), pivot_dfs)

    return final_df

# decriptive statistics

def return_empty_dct(df, var):
    if var not in df.columns: return {}

def year_stats_df(df):
    return_empty_dct(df, "Year")
    s = df["Year"]
    m, M = s.min(), s.max()
    return {"First year": m, "Last year": M, "Period length": M-m+1,
                 "Most productive year": s.mode().min(), 
                 "Average year": s.mean(), "Standard deviation of year": s.std(),
                 "Timespan": str(m)+" : "+str(M)}

def cits_stats_df(df):
    return_empty_dct(df, "Cited by")
    s, df_c = df["Cited by"], df[df["Cited by"]>0]
    return {"Total citations": s.sum(), "Number of cited documents": len(df_c),
            "Proportion of cited documents": len(df_c)/len(df),
            "Highest number of citations": s.max(),
            "Average number of citations": s.mean(), 
            "Average number of citations (cited document)": df_c["Cited by"].mean()}

def occ_stats_df(df, var, sep="; "):
    return_empty_dct(df, var)
    s = df[var]
    no = count_occurrences_by_row(s, sep)
    uo = count_unique_occurrences(s, sep)
    # tukaj bi lahko dodal reference/ključne besede ... na dokument z referencami
    return {"Number of %s" %var.lower(): no, "%s per document" %var: no/len(df),
            "Number of unique %s" %var.lower(): uo,
            "Number of documents per %s" %var.lower(): len(df)/uo}

def n_auth_dist_df(df, var, sep="; ", rm=["", " ", "[No author id available]"]):
    return n_occurrences_by_row(df[var], sep, rm=rm)


def get_add_stats(df, df_p, v, v2=None, items=[], exclude=[], 
                  top=20, top_by="Number of documents", 
                  min_freq=5, cov_prop=0.8, level=1, ord_by_cits=False):

    if v2 is None:
        v2 = v
    computed_later = int(len(items)==0) * int(top_by not in df_p.columns)
    items = select_items(df_p, v, top=top, top_by=top_by, items=items, exclude=exclude, min_freq=min_freq, cov_prop=cov_prop)
    ps = get_performances(df, v2, items, level=level)      
    df_p = pd.merge(df_p, ps, on=v)
    if computed_later: # computed above
        df_p = df_p.sort_values(top_by, ascending=False).head(top)
    if ord_by_cits:
        return order_by_cits(df_p)
    return df_p
    
        
#self.df = miscbib.doc_name(self.df)
#self.df["Author et al."] = self.df["Authors"].map(miscbib.author_to_etal)

def get_doc_names(df):
    if utilsbib.is_sublist(["Authors", "Year", "Abbreviated Source Title"], df.columns):
        a = df["Authors"].astype(str).map(lambda x : x.split(",")[0])
        y = df["Year"].astype(str)
        s = df["Abbreviated Source Title"].astype(str)
        if ("Titles" in df.columns) or ("titles" in df.columns):
            t_var = "Titles" if "Titles" in df.columns else "titles"
            tt = df[t_var].astype(str)
            t = tt.map(lambda x: " ".join(x.split(" ")[:2]))
            df["Name"] = a + ", " + y + ", " + s + ", " + t
            df["Full name"] = a + ", " + y + ", " + s + ", " + tt
        df["Short name"] = a + ", " + y + ", " + s
    return df

def get_topic_var(df):
    t_var = [c for c in df.columns if c in ["title", "titles", "Title", "Titles"]][0]
    if "cleaned abstract" in df.columns:
        a_var = "cleaned abstract"
    else:
        a_var = "abstract" if "abstract" in df.columns else "Abstract"
    ak_var = utilsbib.first_element(["author keywords", "authors keywords", "Author Keywords", "Authors Keywords"], df.columns)
    ik_var = utilsbib.first_element(["indexed keywords", "index keywords", "Indexed Keywords", "Index Keywords"], df.columns)
    
    df["topic"] = df[t_var].astype(str).str.lower() +\
        "\n" +  df[a_var].astype(str).str.lower() +\
            "\n" + df[ak_var].astype(str).str.lower()
    df["topic 2"] = df["topic"] + "\n" + df[ik_var].astype(str).str.lower()
    return df

    
def author_to_etal(aut):
    aut_s = str(aut).split(",")
    if len(aut_s) > 1:
        return aut_s[0] + " et. al" 
    return aut_s[0]

def get_sources_abb_dict(df):
    if utilsbib.is_sublist(["Source title", "Abbreviated Source Title"], df.columns):
        return utilsbib.dict_from_two_columns(df, "Source title", "Abbreviated Source Title")
    return None

def get_sources_issn_dict(df):
    if utilsbib.is_sublist(["Source title", "ISSN"], df.columns):
        return utilsbib.dict_from_two_columns(df, "Source title", "ISSN")
    return None

def get_auth_dict_from_row(row, d={}, db="scopus"):
    if db == "scopus":
        au, au_id = row[1]["Authors"], row[1]["Author(s) ID"]
    elif db == "wos":
        au, au_id = row[1]["Authors"], row[1]["Authors"]
    if au_id != au_id or au_id == "[No author id available]":
        d0 = {}
    else:
        if db == "scopus":
            d0 = dict(zip(*(au_id.split("; "), au.split("; "))))
        elif db == "wos":
            d0 = dict(zip(*(au_id.split("; ")[:-1], au.split("; ")[:-1])))
    return d0

def get_dict_of_authors(df, db="scopus"):   
    d = {}
    for row in df.iterrows():
        d0 = get_auth_dict_from_row(row, db=db)
        d.update(d0)
    return d

def select_items_by(df, name_col, top=10, top_by="Number of documents", 
                    freq_col="Number of documents", min_freq=1, 
                    prop_col="", cov_prop=1., exclude=[]):
    if top > 0 and top_by in df.columns:
        items = df.sort_values(top_by, ascending=False)[name_col].tolist()
        items = [it for it in items if it not in exclude]
        items = items[:top]
    elif min_freq > 1:
        items = df[df[freq_col]>=min_freq][name_col].tolist()
    elif cov_prop < 1:
        items = df[df[prop_col]>=cov_prop][name_col].tolist()
    else:
        items = df[name_col].tolist()
    print(len(items))
    return items
  
def select_items(df, name_col, items=[], exclude=[], 
                 top=10, top_by="Number of documents", 
                 freq_col="Number of documents", min_freq=1, cov_prop=1., 
                 prop_col=""): # TO DO: uredi prop_col
    if len(items) == 0:
        items = select_items_by(df, name_col, top=top, top_by=top_by, 
                                freq_col=freq_col, min_freq=min_freq, 
                                cov_prop=cov_prop, prop_col=prop_col, exclude=exclude)            
    items = [it for it in items if it not in exclude]
    return items

def ind_from_items(df, name_col, items, match="single", add_char=""):
    df_out = pd.DataFrame(columns=items)
    if match == "single":
        for it in items:
            df_out[it] = (df[name_col]==it).astype(int)
    elif match == "multiple":
        for it in items:
            df_out[it] = (df[name_col].astype(str).str.contains(it+add_char).astype(int))
    return df_out


# keywords replacers
    

def replacer_to_bibliometrix():
    pass

def replacer_from_bibliometrix():
    pass


"""
def ind_items(df, name_col, items=[], exclude=[], min_freq=1, cov_prop=1.):
    items = select_items(df, name_col, unit_freq="doc", items=items, exclude=exclude, 
                         min_freq=min_freq, cov_prop=cov_prop)
    df_out = pd.DataFrame(columns=items)
    for it in items:
        df_out[it] = (df[name_col]==it).astype(int)
    return df_out
"""    


# some dictionaries
freqs_rename_dct = {"f": "Number of documents", "f0": "Proportion of documents",
                    "percentage": "Percentage of documents",
                    "count": "Number of documents", 
                    "proportion": "Proportion of documents"}


fd = os.path.dirname(__file__)        
df_countries = pd.read_excel(fd + "\\additional files\\countries.xlsx")
domain_dct = df_countries.set_index("Internet domain").to_dict()["Name"]
c_off_dct = df_countries.set_index("Official name").to_dict()["Name"]
code_dct = df_countries.set_index("Name").to_dict()["Code"]    
    
# functions for analysis of countries

l_countries = list(df_countries["Name"])
eu_countries = list(df_countries[df_countries["EU"]==1]["Name"])

def correct_country_name(s):
    if s != s:
        return ""
    if s in l_countries:
        return s
    elif s in c_off_dct:
        return c_off_dct[s]
    else:
        return ""

def split_ca(s):
    try:
        ca, long_aff = s.split("; ")[:2]
    except:
        return np.nan, np.nan, np.nan
    la = long_aff.split(", ")
    aff, country = la[0], la[-1]
    return ca, aff, country

def parse_mail(s):
    if "@" in s:
        dom = s.split("@")[1].split(" ")[0].split(".")[-1]
        if dom in domain_dct:
            return domain_dct[dom]
        else:
            return np.nan
    return np.nan
    
def get_ca_country_scopus(s, l_countries=l_countries):
    ca, aff, country = split_ca(s)
    if country not in l_countries:
        if country == country:
            cc = [c for c in l_countries if c in country]
            if len(cc) == 1:
                country = cc[0]
            else:
                country = parse_mail(s)
        if country != country and s == s: # could not parse
            cc = [c for c in l_countries if c in s]
            if len(cc) == 1:
                country = cc[0]
            else:
                country = parse_mail(s)
    return country

def get_ca_coutry_wos(s, l_countries=l_countries):
    if s != s:
        return np.nan
    if "USA" in s:
        return "United States"
    if ("England" in s) or ("Scotland" in s) or ("Wales" in s) or \
        ("Northern Ireland" in s) or ("Great Britain" in s):
        return "United Kingdom"
    country = s.split(", ")[-1].replace(".", "")
    if country not in l_countries:
        if country == country:
            try:
                country = c_off_dct[country]
            except:
                print(country)
            if country not in l_countries:
                return np.nan
    return country

def get_ca_country(s, db, l_countries=l_countries):
    if db.lower() == "scopus":
        return get_ca_country_scopus(s, l_countries=l_countries)
    elif db.lower() == "wos":
        return get_ca_coutry_wos(s, l_countries=l_countries)
    
def add_ca_country_df(df, db): # razmisli, kako bi dodal l_countries
    if (db.lower() == "scopus") and ("Correspondence Address" in df.columns):
        df["CA Country"] = df["Correspondence Address"].map(get_ca_country_scopus)
    else:
        print("Not supported yet")
    return df

# all countries and affiliations

def extract_affs_and_countries_scopus(df):
    all_affs, all_affs_un, all_cnts, all_cnts_un = [], [], [], []
    l_cnts = []
    for case in df["Authors with affiliations"]:
        if case == case:
            aas = case.split("; ")
            aff_tmp, c_tmp = [], []
            for aa in aas:
                affc = "".join(aa.split("., ")[1:])
                aff = "".join(affc.split(", "))
                c = correct_country_name(affc.split(", ")[-1])
                if len(aff):
                    aff_tmp.append(aff)
                if len(c):
                    c_tmp.append(c)
            aff_var = "; ".join(aff_tmp)
            aff_un_var = "; ".join(list(set(aff_tmp)))
            c_var = "; ".join(c_tmp)
            c_un_var = "; ".join(list(set(c_tmp)))
            l_cnts += c_tmp
        else:
            aff_var, aff_un_var, c_var, c_un_var = "", "", "", ""
        all_affs.append(aff_var)
        all_affs_un.append(aff_un_var)
        all_cnts.append(c_var.replace(" ; ", ""))
        all_cnts_un.append(c_un_var.replace(" ; ", ""))
    df["Countries"], df["Countries un"] = all_cnts, all_cnts_un
    df["Affiliations"], df["Affiliations un"] = all_affs, all_affs_un
    l_cnts = list(set(l_cnts))
    return df, l_cnts

def get_country_collaboration_df(df):
    if "CA Country" not in df.columns:
        return pd.DataFrame()
    df["SCP"] = (df["CA Country"] == df["Countries un"]).astype(int)
                
    df_country_collab = pd.crosstab(df["CA Country"], df["SCP"])
    df_country_collab.columns=["MCP","SCP"]
    df_country_collab["Number of documents"] = df_country_collab.sum(axis=1)
    df_country_collab["MCP ratio"] \
        = df_country_collab["MCP"] / df_country_collab["Number of documents"]
    df_country_collab = df_country_collab.sort_values(
        "Number of documents", ascending=False).reset_index()
    return df_country_collab.rename(columns={
        "CA Country": "Country of corresponding author"})

def extract_affs_and_countries(df, db):
    if db.lower() == "scopus":
        return extract_affs_and_countries_scopus(df)
    return df, None

def get_top_gl_cit_docs(df, top=10, limit_to=None, add_vars=[], rm_vars=[]):
        base_vars=["Authors", "Author et. al", "Titles", "titles",
                   "Source title", "Cited by", "Year", "Document Type"]
        vrs = [v for v in base_vars if v not in rm_vars] + add_vars
        df = df[[v for v in vrs if v in df.columns]]
        if limit_to is not None:
            df = df[df["Document Type"].isin(limit_to)]
        return df.sort_values("Cited by", ascending=False)[:top]


def which_keywords(df, which="ak"):
    if which == "ak":
        return utilsbib.first_element(["author keywords", "authors keywords", "Author Keywords", "Authors Keywords"], df.columns)
    elif which == "ik":
        return utilsbib.first_element(["indexed keywords", "index keywords", "Indexed Keywords", "Index Keywords"], df.columns)
    elif which == "aik":
        print("Not yet supported")
        return None

# functions for occurrence analysis

def count_occurrences_by_row(s, sep):
    return s.dropna().astype(str).str.split(sep).map(len).sum()
        
def extract_occurrences_by_row(s, sep, rm=["", " "]):
    occ = []
    for index, value in s.items():
        if value != value:
            occ.append([])
        else:
            t = value.split(sep)
            for r in rm:
                while r in t:
                    t.remove(r)
            occ.append(t)
    return occ

def n_occurrences_by_row(s, sep, rm=["", " "]):
    return list(map(len, extract_occurrences_by_row(s, sep, rm=rm)))

def distr_n_occurrences(s, sep, rm=["", " "]):
    return utilsbib.freqs_from_list(n_occurrences_by_row(s, sep, rm=rm), order=2)

def extract_occurrences(s, sep, rm=["", " "]):
    return utilsbib.unnest(extract_occurrences_by_row(s, sep, rm=rm))

def extract_unique_occurrences(s, sep, rm=["", " "]):
    return list(set(extract_occurrences(s, sep, rm=rm)))

def count_unique_occurrences(s, sep, rm=["", " "]):
    return len(extract_unique_occurrences(s, sep, rm=rm))

def count_occurrences(s, sep, rm=["", " "], order=2):
    return utilsbib.freqs_from_list(extract_occurrences(s, sep, rm=rm), order=order)

def fractional_count_occurrences(s, sep, rm=["", " "], sort=True):
    ocr = extract_occurrences_by_row(s, "; ", rm=rm)
    d = []
    for oc in ocr:
        n = len(oc)
        for o in oc:
            d.append([o, 1/n])  
    fc = pd.DataFrame(d).groupby(0).agg(sum).reset_index()
    fc.columns = ["item", "fractional count"]
    if sort:
        fc = fc.sort_values("fractional count", ascending=False)
    return fc

def count_occurrences_df(s, sep, rm=["", " "], sort_by="count", add_relative=True, order=2, relative_rank=True):
    df = pd.DataFrame(count_occurrences(s, sep, rm=rm, order=order), columns=["item", "count"])
    if add_relative:
        n = len(df)
        df["proportion"] = df["count"] / n
    if relative_rank:
        df["Relative rank"] = df["count"].rank(pct = True) 
    df = df.rename(columns=freqs_rename_dct)
    return df

def both_counts_occurrences(s, sep, rm=["", " "], sort_by="count", add_relative=True, relative_rank=True):
    c = pd.DataFrame(count_occurrences(s, sep, rm=rm), columns=["item", "count"])
    fc = fractional_count_occurrences(s, sep, rm=rm)
    fcc = pd.merge(c, fc, on="item")
    if add_relative:
        n = len(s)
        fcc["proportion"], fcc["fractional proportion"] = fcc["count"]/n, fcc["fractional count"]/n
    if relative_rank:
        fcc["Relative rank"] = fcc["count"].rank(pct = True) 
    fcc = fcc.sort_values(sort_by, ascending=False)
    return fcc

def get_collaboration_index(s, sep, rm=["", " "]):
    o = extract_occurrences_by_row(s, sep, rm=rm)
    ma_docs = [oo for oo in o if len(oo) > 1]
    ma_auth = utilsbib.unnest([oo for oo in o if len(oo) > 1])
    if len(ma_docs):
        return len(ma_auth) / len(ma_docs)
    return np.nan

def collab_stats(s, sep, rm=["", " "]):
    f, n = dict(distr_n_occurrences(s, sep, rm=rm)), len(s)
    nm = sum(f[k] for k in f if k > 1)
    return {"Single author documents": f[1], 
            "Proportion of single author documents": f[1]/n, 
            "Multi author documents": nm, 
            "Proportion of multi author documents": nm/n,
            "Co-authors per document": count_occurrences_by_row(s, sep) / n,
            "Collaboration index": get_collaboration_index(s, sep, rm=rm)}

def count_and_store_0(df, var, sep="; ", name=None):
    if var not in df.columns:
        return df
    if name is None:
        name = "Number of " + var
    if name not in df.columns:
        df[name] = df[var].map(lambda x: utilsbib.split_and_count(x, sep=sep))
    return df
    
def count_and_store(df, vrs): # vrs should be list of tuples (var, sep, name)
    for v in vrs:
        var, sep, name = v
        try:
            df = count_and_store_0(df, var.lower(), sep=sep, name=name)
        except:
            df = count_and_store_0(df, var, sep=sep, name=name)
    return df

# some aggregation functions



# scientific production

def get_production(df, fill_empty=True, rng=None, cut_year=None, exclude_last="smart"):
    n, cits = "Number of documents", "Cited by"
    
    if "Year" not in df.columns or "Cited by" not in df.columns:
        return None
    
    # Ensure 'Year' is numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)
    
    # Calculate yearly production and citation totals
    yp = df["Year"].value_counts()
    cy = df[["Year", "Cited by"]].groupby("Year").agg("sum").sort_index()
    
    # Combine into a single DataFrame
    prod = pd.concat([yp, cy], axis=1)
    prod.columns = [n, cits]  # Rename columns for clarity

    # Fill in missing years with zeros if requested
    if fill_empty:
        all_years = range(prod.index.min(), prod.index.max() + 1)
        prod = prod.reindex(all_years, fill_value=0)
    
    # Apply the range filter if specified
    if rng is not None:  # rng - tuple (min, max)
        prod = prod.loc[range(*rng)]
    elif cut_year is not None:
        # Calculate the sum of the years before the cut year
        before = prod.loc[:cut_year].sum(numeric_only=True)  # Ensure numeric operations
        prod = prod.loc[cut_year:]
        # Update the last row with summed values from before cut_year
        for col in prod.columns:
            if col in before:
                prod.at[prod.index[-1], col] = before[col]
    
        
    # Handle exclusion of the last row based on a condition
    if exclude_last == "smart":
        if prod.loc[prod.index.max()][n] / prod.loc[prod.index.max() - 1][n] < 0.75:
            exclude_last = True
        else:
            exclude_last = False
    
    if exclude_last:
        prod = prod.iloc[:-1]
    

    if cut_year is not None:
        prod = prod.rename(index={prod.index[-1]: "before %d" % cut_year})
        last_row = prod.iloc[-1:]
        remaining_df = prod.iloc[:-1]
        prod = pd.concat([last_row, remaining_df])

    # Add cumulative columns
    #prod = prod.sort_index()
    prod["Cumulative number of documents"] = prod[n].cumsum()
    prod["Cumulative number of citations"] = prod[cits].cumsum()
    prod["Average citations per document"] = prod[cits] / prod[n]
    prod["Average citations per document"] = prod["Average citations per document"].replace({np.nan: 0})

    
    # Reset index to make 'Year' a column again
    prod = prod.reset_index().rename(columns={"index": "Year"})
    
    return prod






def percentage_growth(prod_df, exclude_last=True): # ni fleksibilna, vrne za točno določena leta
    prod_df.pct_change()["Number of documents"]

    fy, ly = prod_df["Year"].min(), prod_df["Year"].max()
    l = ly-fy
    
    if l < 10:
        print("Too short period")
        return {}

    fy_i, ly_i = prod_df["Year"].argmin(), prod_df["Year"].argmax() - int(exclude_last)
    d0, d1, d5, d10, dn = [prod_df.at[i, "Number of documents"] for i in [fy_i, ly_i-1, ly_i-5, ly_i-10, ly_i]] 
    g, g1, g5, g10 = (dn/d0)**(1/(l-1))-1, dn/d1-1, (dn/d5)**(1/4)-1, (dn/d10)**(1/9)-1
    
    return {"Average document growth (overall) [%]": g*100, "Document growth last year [%]": g1*100, "Document growth last 5 years [%]": g5*100, "Document growth last 10 years [%]": g10*100}

def get_productions_ind(ind_df, s, fill_empty=True): # tipično s=df["Year"] (od nekega drugega df)
    
    merge_df = pd.concat([ind_df, s], axis=1)

    sers = []
    for c in ind_df.columns:
        ct = pd.crosstab(merge_df[c], merge_df["Year"]).loc[1]
        ct.name = c
        sers.append(ct)
        ctc = ct.cumsum()
        ctc.name = "cum " + c
        sers.append(ctc)

    prod = pd.concat(sers, axis=1)

    if fill_empty:
        for y in range(prod.index.min(), prod.index.max()):
            if y not in prod.index:
                prod.at[y] = 0
    
        
    prod = prod.sort_index().reset_index()
    return prod

# text processing functions
    
def translate_kw(s, d):
    if s != s:
        return ""
    words = s.split(";")
    words = [w.strip() for w in words if w.strip()]
    if len(words[-1]) == 0:
        words = words[:-1]
    t_words = []
    for w in words:
        if w in d:
            t_words.append(d[w])
        else:
            t_words.append(w)
    return "; ".join(t_words)

# additional info for scopus data (info on souces has to be dowloaded from scopus webpage)


def add_source_info(df, df_sources, source_var="Source Title",
                    source_issn_var_1 = "ISSN", source_issn_var_2 = "EISSN", 
                    codes_var =	"All Science Journal Classification Codes (ASJC)"):
    codes_list = []
    

    for _, row in df.iterrows():
        issn = row["ISSN"]
        source_title = row["Source title"]

        matching_row = df_sources[(df_sources[source_issn_var_1] == issn) | (df_sources[source_issn_var_2] == issn) | (df_sources[source_var] == source_title)]

        if not matching_row.empty:
            codes = matching_row.iloc[0][codes_var]
            codes_list.append(codes)
        else:
            codes_list.append(np.nan)



    return pd.DataFrame(codes_list, columns=["Codes"])

"""
def add_source_info(df, df_sources, source_var="Source Title (Medline-sourced journals are indicated in Green)",
                    source_issn_var_1 = "Print-ISSN", source_issn_var_2 = "E-ISSN", 
                    codes_var =	"All Science Journal Classification Codes (ASJC)"):
    

    issns = df["ISSN"].unique()

    m1, m2 = [], []
    
    for issn in issns:
        if issn in df_sources[source_issn_var_1].values:
            m1.append(issn) 
        elif issn in df_sources[source_issn_var_2].values:
            m2.append(issn)
            
    d1 = df_sources[df_sources[source_issn_var_1].isin(m1)][[source_issn_var_1, codes_var]]
    d2 = df_sources[df_sources[source_issn_var_2].isin(m2)][[source_issn_var_2, codes_var]]
    
    nc = len(d1.columns)-1
    
    l=[]
    
    dd1 = d1.set_index(source_issn_var_1).T.to_dict("list")
    dd2 = d2.set_index(source_issn_var_2).T.to_dict("list")
    
    for issn in df["ISSN"].values:
        if issn in dd1:
            l.append(dd1[issn])
        elif issn in d2:
            l.append(dd2[issn])
        else:
            l.append([np.nan]*nc)
            
    df_add = pd.DataFrame(l, columns=["Codes"])
    df_add["Codes"] += " "

    return df_add 
"""

def translate_scopus_codes(df):
    df_scopus_codes = pd.read_excel(fd + "\\additional files\\scopus subject area codes.xlsx")
    for (lvl, name) in [("level 3", "Fields"), ("level 2", "Areas"), ("level 1", "Sciences")]:
        d = df_scopus_codes.set_index("code").to_dict()[lvl]
        d = {str(k):v for k,v in d.items()}
        df[name] = df["Codes"].apply(lambda x: translate_kw(x, d))
    return df
    

# performance indicators (computation fucntions)

def h_index_l(cit, alpha=1):
    return sum(x >= alpha * (i + 1)
               for i, x in enumerate(sorted(list(cit), reverse=True)))
    
def g_index_l(cit):
    nd = len(cit)
    cit.sort(reverse=True)
    cumcit = np.cumsum(cit)
    sqs = np.arange(1, nd+1)**2
    ind = cumcit - sqs
    res = [i for i, x in enumerate(ind) if x<0]
    if res == []:
        return nd
    else:
        return res[0]
    
def hg_index_l(cits):
    return np.sqrt(h_index_l(cits)*g_index_l(cits))

def c_index_l(cits, cs=[5, 10, 20, 50, 100]):
    return dict([(c, sum(np.array(cits) >= c)) for c in cs])
    
def tapered_h_l(cits):
    cits, h = map(int, cits), 0
    for i, cit in enumerate(cits):
        k = min(i+1, cit)
        h += k / (2*i+1)
        h += sum([1/(2*i+1) for i in range(i+1, cit)])
    return h
    
def chi_index_l(cits):
    cits.sort(reverse=True)
    return np.sqrt(max([(i+1)*cits[i] for i in range(len(cits))]))

# performance indicators (on DataFrame)
    
def total_cit_df(df):
    if "Cited by" in df.columns:
        return df["Cited by"].sum()
    return None

def av_year_df(df):
    if "Year" in df.columns:
        return df["Year"].mean()
    return None

def percentile_year(df, perc=0.5):
    if "Year" in df.columns:
        return df["Year"].quantile(q=perc)
    return None
    
def first_year(df):
    if "Year" in df.columns:
        return df["Year"].min()
    return None
    
def last_year(df):
    if "Year" in df.columns:
        return df["Year"].max()
    return None


def h_index_df(df, alpha=1):
    if "Cited by" in df.columns:
        return h_index_l(df["Cited by"].tolist(), alpha=alpha)
    return None
  
def g_index_df(df):
    if "Cited by" in df.columns:
        return g_index_l(df["Cited by"].tolist())
    return None

def hg_index_df(df):
    if "Cited by" in df.columns:
        return hg_index_l(df["Cited by"].tolist())
    return None

def c_index_df(df, cs=[5, 10, 20, 50, 100]):
    if "Cited by" in df.columns:
        return c_index_l(df["Cited by"].tolist(), cs=cs) # returns dictionary
    return dict([(c, None) for c in cs])

def tapered_h_df(df):
    if "Cited by" in df.columns:
        return tapered_h_l(df["Cited by"].tolist())
    return None
    
def chi_index_df(df):
    if "Cited by" in df.columns:
        return chi_index_l(df["Cited by"].tolist())
    return None

def get_perf_ind(df, name=None, level=1, name_var="Name", measure=entropy):
    if name is not None:
        p = [(name_var, name)]
    else:
        p = []
    if level >= 0: # core performance indicators
        tot_cit = total_cit_df(df)
        h_index = h_index_df(df)
        av_year = av_year_df(df)
        p += [("Total number of citations", tot_cit), ("H-index", h_index), 
              ("Average year of publication", av_year)]
    if level >= 1: # advanced performance indicators
        g_index = g_index_df(df)
        c_index = c_index_df(df)
        p += [("G-index", g_index)]
        p += [("C%d" %i, c_index[i]) for i in [5, 10, 20, 50, 100]]
        p += [("First year", first_year(df)),
              ("Q1 year", percentile_year(df, perc=0.25)), 
              ("median year", percentile_year(df, perc=0.5)),
              ("Q3 year", percentile_year(df, perc=0.75)),
              ("Last year", last_year(df))]
    if level >= 2: # interdisciplinarity
        if "Cited Health Sciences" in df.columns:
            cs = df[[c for c in df.columns if "Cited" in c if c != "Cited by"]].sum()
            if measure == entropy:
                p += [("Interdisciplinarity", measure(cs)/np.log(5))]
            else:
                p += [("Interdisciplinarity", measure(cs))]
            p += list(zip(cs.index, cs))
            
    if level >= 3: # specific, rarely used indicators
        pass
    return p

def get_performances(df, name_var, items, search="exact", level=1):
    ps = []
    for it in items:
        if search == "exact":
            df_s = df[df[name_var]==it]
        elif search == "substring":
            df_s = df[df[name_var].astype(str).str.contains(it)]
        p = get_perf_ind(df_s, it, level=level, name_var=name_var)
        ps.append(p)
    return pd.DataFrame(list(map(dict, ps)))

def order_by_cits(df):
    c_var, n_var = "Total number of citations", "Number of documents"
    if c_var in df.columns:
        df = df.sort_values(c_var, ascending=False)
        if n_var in df.columns:
            df["Average citations per document"] = df[c_var] / df[n_var]
    return df

def round_properly(x, n_dig=3):
    if x in ["Number of documents", "Total number of citations",
             "H-index", "G-index", "C5", "C10", "C20", "C50", "C100"]:
        return 0
    return n_dig


# clustering based on indicator dataframes

def mds_from_dfs(dfs, what="variables", metric="cosine", nc=2, groups=[]):
    df = pd.concat(dfs, axis=1)
    labels = df.columns
    if len(groups) == 0:
        groups = ["group %d" % i for i in range(1, len(dfs)+1)]
    df_mds = utilsbib.mds_from_df(df, nc=nc, what=what, metric=metric)
    gr = []
    for i in range(len(dfs)):
        gr += [groups[i]] * len(dfs[i].columns)
    df_mds["group"], df_mds["label"] = gr, labels
    return df_mds


# distributions
    
def get_cnd_dist(df, f_var, y_var, items=[], exclude=[], top=0, kind=None, sep="; "):
    if kind is None:
        kind = 2
        if y_var in ["Source title", "Document Type"]:
            kind = 1
        
    if len(items) == 0:
        if kind == 1:
            items = df[f_var].value_counts().index
        else:
            items = count_occurrences_df(df[f_var], sep)["item"]
        if top > 0:
            items = items[:top]
    if len(exclude):
        items = [it for it in items if it not in exclude]
    d = {}
    if kind == 1:
        for it in items:
            d[it] = df[df[f_var]==it][y_var].values
    else:
        for it in items:
            d[it] = df[df[f_var].astype(str).str.contains(it)][y_var].values
    return d


# images

class BibImage:
    
    def __init__(self, path, name=None, caption=None, id_=None, core=True, fig=None):
        self.path, self.name, self.caption, self.fig = path, name, caption, fig
        self.heading = name.capitalize() if name is not None else ""
        self.id_, self.core = id_, core
    
    def show(self):
        from PIL import Image                                                                                
        img = Image.open(self.path + ".png")
        img.show() 

# REFERENCES

def count_self_ref_sources(df):
    if "References" in df.columns and "Source title" in df.columns:
        l_out = []
        for index, row in df.iterrows():
            l, s = str(row["References"]), row["Source title"]
            l_out.append(l.count(s))
        df["Self referencing sources"] = l_out
    return df


# Interdisciplinarity


def get_cited_sources_scopus(lr):
    sources = []
    for r in lr.split("; "):
        s = extract_source_scopus(r)
        if s == s:
            sources.append(s)
    return sources

def prop_unique_source_occ_scopus(lr):
    cs = get_cited_sources_scopus(lr)
    return len(set(cs))/len(cs)

def get_cited_codes_scopus(lr, d_codes):
    cs = get_cited_sources_scopus(lr)
    return "; ".join([d_codes[c][:-1] for c in cs if c in d_codes])

def number_od_uniqe_sciences_occ_scopus(lr, d_codes):
    cc = get_cited_codes_scopus(lr, d_codes)
    cc_l = cc.split("; ")
    return len(cc_l), len(set(cc_l))

def get_cited_sciences_scopus(lr, d_codes, d1):
    cc = get_cited_codes_scopus(lr, d_codes)
    if ";;" in cc:
        cc = cc.replace(";;", ";")
    cc_l = cc.split("; ")
    if len(cc_l) <= 1:
        return []
    cc_l = [c.replace(";", "") for c in cc_l]
    return [d1[int(c)] for c in cc_l if int(c) in d1] # kode 3330 ni v seznamu

def count_cited_sciences_scopus(lr, d_codes, d1):
    if lr == lr:
        all_sc = ["Multidisciplinary", "Health Sciences", "Life Sciences", "Social Sciences", "Physical Sciences"]
        csc = get_cited_sciences_scopus(lr, d_codes, d1)
        return [csc.count(a) for a in all_sc]
    return [0] * 5



def inter_disc_scopus(lr, d_codes, d1, measure=entropy):
    if lr == lr:
        return measure(count_cited_sciences_scopus(lr, d_codes, d1))
    return np.nan


#df_scopus_codes = pd.read_excel(fd+"\\additional files\\scopus subject area codes.xlsx")
#df_scopus_codes["code"] = df_scopus_codes["code"].astype(str)
#d1, d2, d3 = [df_scopus_codes.set_index("code").to_dict()[f"level {l}"] for l in range(1,4)]

def add_interdisciplinarity_scopus_df(df, d_codes, d1, measure=entropy, add_counts=True):
    if "References" not in df.columns:
        print("No references given")
        return df
    df["Interdisciplinarity"] = df["References"].map(lambda x: inter_disc_scopus(x, d_codes, d1, measure=measure)) / np.log(5)
    if add_counts:
        df_int = pd.DataFrame(df["References"].map(lambda x: count_cited_sciences_scopus(x, d_codes, d1)).tolist(), 
                              columns=["Cited " + c for c in ["Multidisciplinary", "Health Sciences", "Life Sciences",  "Social Sciences", "Physical Sciences"]])
        df = pd.concat([df, df_int], axis=1)
    
    return df





# speroscopy
    
def extract_years(text, min_year=1900, max_year=datetime.datetime.now().year, db="scopus"):
    if (text is None) or (text != text):
        return []
    #nn = [int(s) for s in re.findall(r"\b\d+\b", text)]
    if db == "scopus":
        nn = [int(s) for s in re.findall(r"\(\s*\+?(-?\d+)\s*\)", text)]
    elif db == "wos":
        nn = [int(s) for s in re.findall(r", *\+?(-?\d+)\s*, ", text)]
    return [n for n in nn if n>=min_year if n<=max_year]

def get_spectroscopy(df,  min_year=1900, max_year=datetime.datetime.now().year, db="scopus"):
    years = []
    if "References" not in df.columns:
        return years
    else:
        for item, row in df.iterrows():
            y = extract_years(row["References"], min_year=min_year, max_year=max_year, db=db)
            years.append(y)
    return years


def v_range(x):
    if len(x) == 0:
        return np.nan
    return np.max(x)-np.min(x)



def add_stats_from_refs(df, db, min_year=1900, max_year=datetime.datetime.now().year,
                        stats=[("Average year from refs", np.mean), 
                               ("Span of years from refs", v_range)]):
    years = get_spectroscopy(df,  min_year=min_year, max_year=max_year, db=db)
    for s, f in stats:
        df[s] = list(map(f, years))
    return df, years


def extract_authors_scopus(r):
    try:
        as_ = r.split("., ")[:-1]
        return as_, as_[0] + ".", "., ".join(as_) + "."
    except:
        return [], np.nan, np.nan
    

def extract_year_scopus(r):
    try:
        return int(r.split(", (")[-1][:-1])    
    except:
        return np.nan

def extract_source_scopus(r):
    try:
        pattern = r", (?=[A-Z])"
        source = re.split(pattern, r.split("., ")[-1])[1].split(",")[0]
        if re.match(r"^\(\d+\)$", source):
            return np.nan
        return source
    except:
        return np.nan

def extract_title_scopus(r):
    try:
        pattern = r"(?<!\.)\s*,\s*(?=[A-Z])"
        title = re.split(pattern, r)[0].split("., ")[-1].split(", (")[0]
        #title = r.split("., ")[-1].split(", ")[0]
        if re.match(r"^\(\d+\)$", title):
            return np.nan
        return title
    except:
        return ""


def extract_info_refs_scopus(r):
    
    if (len(r) == 0):
        return [], np.nan, np.nan, np.nan, np.nan, np.nan

    authors_list, first_auth, authors = extract_authors_scopus(r)
    year = extract_year_scopus(r)
    title = extract_title_scopus(r)
    source = extract_source_scopus(r)
            
    return authors_list, first_auth, authors, year, title, source


"""
def extract_info_refs_scopus(r):
    try:        
        rr = r.split("., ")
        sep_authors = [x+"." for x in rr[:-1]]
        authors = "., ".join(rr[:-1]) + "."
        year = extract_years(r)[0]
        rrr = rr[-1].split(" ("+str(year)+") ")
        title = rrr[0]
        try:
            source = rrr[1].split(",")[0]
        except:
            source = ""     
    except:
        sep_authors = sep_authors if "sep_authors" in locals() else []
        authors = authors if "authors" in locals() else ""
        year = year if "year" in locals() else np.nan
        title = title if "title" in locals() else ""
        source = source if "source" in locals() else ""
            
    return sep_authors, authors, year, title, source
"""



def prepare_for_trend_plot(df_prod, items=[], exclude=[], name_var=None, 
                           mid_var="median year", period=None, items_per_year=5):
    if period is None:
        period = [1990, datetime.datetime.now().year]
              
    if len(items):
        if len(exclude):
            items = [it for it in items if it not in exclude]
        df_prod = df_prod[df_prod[name_var].isin(items)]
    elif len(exclude):
        items = df_prod[name_var].tolist()
        items = [it for it in items if it not in exclude]
        df_prod = df_prod[df_prod[name_var].isin(items)]
    
    df_prod = df_prod.dropna(subset=[mid_var])
    
    df_prod[mid_var] = df_prod[mid_var].map(np.ceil).map(int)
    ms = sorted(df_prod[mid_var].unique())
    dfs = []
   
    for m in ms:
        if m in range(*period):
            df_tmp = df_prod[df_prod[mid_var] == m].sort_values(
                "Number of documents", ascending=False).head(items_per_year)
            dfs.append(df_tmp)
    return pd.concat(dfs).iloc[::-1].reset_index()


def prepare_for_sankey(df_inds, ks=10, dicts=[], add_ser=None, fn=np.mean, all_pairs=False):
    Xs = []
    if ks is None:
        ks = [len(df_ind.columns) for df_ind in df_inds]
    if type(ks) == int:
        ks = [ks] * len(df_inds)
    Xs = []
    for i, df_ind in enumerate(df_inds):
        Xs.append(df_ind.iloc[:, :ks[i]])
    df_c = pd.concat(Xs, axis=1)    

    color_vals = []
    if add_ser is not None:
        for c in df_c.columns:
            ser = add_ser[df_c[c]==1].dropna()
            color_vals.append(fn(ser))

    labels = sum([list(x.columns) for x in Xs], [])
    groups = sum([[i]*len(x.columns) for i, x in enumerate(Xs)], [])
    ld = {l:i for i, l in enumerate(labels)}
   
    Ps = []
    if all_pairs:
        for i in range(len(df_inds)):
            for j in range(i+1, len(df_inds)):
                p = Xs[i].T.dot(Xs[j])
                Ps.append(p.unstack().reset_index(0).reset_index(0))
    else:
        for i in range(len(df_inds)-1):
            p = Xs[i].T.dot(Xs[i+1])
            Ps.append(p.unstack().reset_index(0).reset_index(0))
        
    P = pd.concat(Ps)
    P.columns = ["source", "target", "value"]
    def remove_eq(s):
        return s.split(" = ")[1] if " = " in s else s
    P["label"] = P["source"].apply(remove_eq)
    P["source"], P["target"] = P["source"].map(ld), P["target"].map(ld)
    
    md = {key:val for d in dicts for key,val in d.items()}
    def rename(s):
        return md[s] if s in md else s
    
    P["label"] = P["label"].map(rename)
    
    return P, [rename(remove_eq(l)) for l in labels], color_vals, ks, groups

# wordcloud funtions




# network analysis

class BibNetwork:
    
    def __init__(self, df_ind, norm_method="association", partition_algorithm=algorithms.louvain,
                 add_default_stats=True, k_docs_per_cluster=5, rename_dict=None):
        self.df_ind = df_ind
        self.n = len(df_ind)
        self.co_df = df_ind.T.dot(df_ind)
        self.co_s_df = utilsbib.normalize_sq_matrix(self.co_df, method=norm_method)
        
        self.co_pairs_df = utilsbib.pairs_from_sym_matrix(self.co_df),
        self.co_s_pairs_df = utilsbib.pairs_from_sym_matrix(self.co_s_df, keep_self=False)
        
        self.net = utilsbib.net_from_mat(self.co_s_df, names=self.co_s_df.index)

        self.size = len(self.net)
        self.nodes = list(self.net.nodes)
       
        self.part0 = partition_algorithm(self.net)
        self.part = self.part0.communities
        self.n_clusters = len(self.part)
        self.cluster_nodes = {i+1:c for i,c in enumerate(self.part)}
        
        # tole je moja izvirna ideja
        self.df_counts_clusters = pd.DataFrame()
        for i in range(self.n_clusters):
            nodes =  self.cluster_nodes[i+1]
            self.df_counts_clusters[f"Cluster {i+1}"] = df_ind[list(nodes)].sum(axis=1)
        self.df_ind_clusters = (self.df_counts_clusters > 0).astype(int)
        self.df_s_counts_clusters = self.df_counts_clusters / self.df_counts_clusters.max(axis=0)
        
        # s tem lahko delamo co-occurence clustrov - na višjem nivoju odnos 
        self.co_clusters = self.df_ind_clusters.T.dot(self.df_ind_clusters)
        self.co_s_clusters = utilsbib.normalize_sq_matrix(self.co_clusters, method=norm_method)
        
        # za vsakega lahko določimo tudi karakteristične dokumente
        self.char_docs = {}
        for i in range(self.n_clusters):
            M = self.df_counts_clusters[f"Cluster {i+1}"].max()
            self.char_docs[i+1] = self.df_counts_clusters[self.df_counts_clusters[f"Cluster {i+1}"]==M].index
        
        # alternativno
        if k_docs_per_cluster is not None:
            self.df_s_counts_clusters_rnk = self.df_s_counts_clusters.rank(ascending=False, method="min")
            self.char_docs_cluster = {c: self.df_s_counts_clusters_rnk.index[self.df_s_counts_clusters_rnk[c]<k_docs_per_cluster] for c in self.df_s_counts_clusters_rnk}
        
        
        self.cluster_sizes = {i:len(c) for i,c in self.cluster_nodes.items()}
        self.partition = {t: c for c, ts in self.cluster_nodes.items() for t in ts}
        self.partition = {n: self.partition[n] for n in self.nodes}

        self.partition_df = pd.DataFrame(self.partition.items(), columns=["Node", "Cluster ID"])
        
        self.clusters_df = pd.DataFrame.from_dict(self.cluster_sizes, orient="index", columns=["size"])
        
        if add_default_stats:
            self.get_clusters_labels()
            self.get_clusters_stats()
            self.clusters_df.columns =  list(map(lambda x: x.replace("_", " "), self.clusters_df.columns.tolist()))
        
        if rename_dict is not None:
            self.net = nx.relabel_nodes(self.net, rename_dict)
    
    def get_clusters_labels(self, labels_join="\n", cut_labels=None):
        if cut_labels is None:
            cut_labels = self.size
        for i in range(self.n_clusters):
            C = self.net.subgraph(self.cluster_nodes[i+1])
            l = [n for n in self.nodes if n in C.nodes][:cut_labels]
            if labels_join is not None:
                self.clusters_df.loc[i+1, "label"] = labels_join.join(l)
        
    def get_clusters_stats(self, fun_props_1=["group_degree_centrality"],
                           fun_props_2=["density"], rankings=True):
        
        if self.n_clusters == 1:
            print("Just one cluster")
            return None
        for i in range(self.n_clusters):
            C = self.net.subgraph(self.cluster_nodes[i+1])
            for f1 in fun_props_1: # properties that require subgraph and original graph
                f = getattr(nx, f1)
                self.clusters_df.loc[i+1, f1] = f(self.net, C)
            for f2 in fun_props_2:
                f = getattr(nx, f2)
                self.clusters_df.loc[i+1, f2] = f(C)
        if rankings:
            for p in fun_props_1 + fun_props_2:
                self.clusters_df["rank " + p] = self.clusters_df[p].rank()
                
        
    def get_docs_stats(self, additional_df, add_long_doc_name=True):
        if ("Cited by" not in additional_df) or ("Year" not in additional_df) or (len(additional_df) != self.n):
            print("No relevant data")
            return None
        new_rows = []
        for i, c in enumerate(self.df_ind_clusters.columns):
            df_tmp = additional_df[self.df_ind_clusters[c]==1]
            c_docs = ""
            if "Short name" in additional_df.columns:
                c_docs = "\n".join(additional_df.loc[self.char_docs[i+1], "Short name"].tolist())
            r = [len(df_tmp), df_tmp["Cited by"].sum(), df_tmp["Year"].mean(), c_docs]
            if ("Full name" in additional_df.columns) and add_long_doc_name:
                c_docs_l = "\n".join(additional_df.loc[self.char_docs[i+1], "Full name"].tolist())
                r.append(c_docs_l)

            new_rows.append(r)

        cols = ["Number of documents", "Total number of citations", "Average year", "Representative documents"]
        if add_long_doc_name:
            cols.append("Representative documents (long)")
        new_df = pd.DataFrame(new_rows, columns=cols)
        new_df.index += 1
        self.clusters_df = pd.concat([self.clusters_df, new_df], axis=1)

    def get_vectors(self, df_add, name=None):
        self.vectors, self.vectors_pajek = {}, {}
        if name is None:
            name = df_add.columns[0]
        df_add = df_add[df_add[name].isin(self.nodes)] 
        df_add = df_add.set_index(name)
        df_add = df_add.reindex(self.nodes).reset_index()
        for c in df_add:
            self.vectors[c] = df_add[c]
            self.vectors_pajek[c] =  "*Vertices %d\n" % len(self.nodes) +\
            "\n".join([str(x) for x in self.vectors[c].values])

    
    def to_pajek(self, f_name, save_vectors="all"):
        nx.write_pajek(self.net, f_name + ".net")
        
        self.pajek_partition = "*Vertices %d\n" % len(self.nodes) +\
            "\n".join([str(x) for x in self.partition.values()])
            
        with open(f_name + ".clu", "wt") as f:
            f.write(self.pajek_partition)
        
        if not hasattr(self, "vectors_pajek"):
            return None
        if save_vectors == "all":
            vectors = self.vectors_pajek
        else:
            vectors = [v for v in save_vectors if v in self.vectors_pajek.keys()]
        for v in vectors:
            with open(f_name + " - %s.vec" % v, "wt") as f:
                f.write(self.vectors_pajek[v])


class TwoConceptsBib(utilsbib.TwoConcepts):
    
    def get_coocurence_net(self, use_ind=True): # izvirna ideja - isti koncept, a določen ne na podlagi dokumentov
        d12 = self.d12_ind if use_ind else self.d12
        self.co_net = BibNetwork(d12)
        
    def get_coupling_net(self, use_ind):
        d21 = self.d21_ind if use_ind else self.d21
        self.coup_net = BibNetwork(d21)