# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 13:17:51 2022

@author: Lan.Umek
"""

import os
import pandas as pd
import itertools
import numpy as np

fd = os.path.dirname(__file__)   

import networkx as nx

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lf0 = WordNetLemmatizer()
lf = lf0.lemmatize

from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from networkx.algorithms import community as cmt
import datetime
import re

# for Lotka's law
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score, mean_squared_error


def unnest(lst):
    return list(itertools.chain(*lst))

def sort_matrix_elements(mat, reverse=True, remove_trivial=False):
    pairs = []
    for index, row in mat.iterrows():
        for r in row.iteritems():
            if r[0] == index and remove_trivial:
                continue
            pairs.append([r[1], r[0], index])
        
    pairs = sorted(pairs, reverse=reverse)
    pairs = pd.DataFrame(pairs, columns=["value", "item 1", "item 2"])
    return pairs[["item 1", "item 2", "value"]]

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def make_folders(folders):
    for folder in folders:
        make_folder(folder)
        
def is_float(s):
    try:
        float(s)
        return True
    except:
        return False
    
def is_int(s):
    if s == int(s):
        return True
    return False

def smart_round(s, ndig=3):
    try:
        if s != s:
            return s
        if is_int(s):
            return int(float(s))
        else:
            return np.round(s, decimals=ndig)
    except:
        return s
        
def is_sublist(l1, l2):
    return set(l1) <= set(l2)

def first_element(l, s):
    for element in l:
        if element in s:
            return element
    return None


def check_balance(s):
    if type(s) != "str":
        return True
    # based on https://www.geeksforgeeks.org/check-for-balanced-parentheses-in-python/
    open_list, close_list = "[{(", "]})" 
    stack = []
    for i in s:
        if i in open_list:
            stack.append(i)
        elif i in close_list:
            pos = close_list.index(i)
            if ((len(stack) > 0) and
                (open_list[pos] == stack[len(stack)-1])):
                stack.pop()
            else:
                return False
    return True  if len(stack) == 0 else False

def count_words_string(s):
    if s != s:
        return None
    return len(str(s).split())


def check_missing_values(df, columns=None):
    """
    Returns the number and proportion of missing values for each column in a given list of columns in a pandas DataFrame.
    
    Parameters:
        df (pandas.DataFrame): The pandas DataFrame to check for missing values.
        columns (list): A list of column names to check for missing values.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the number and proportion of missing values for each column in the given list of columns, as well as a new column indicating the quality of each column based on the number of missing values.
    """
    if columns is None:
        columns = df.columns
    else:
        columns = [c for c in columns if c in df.columns]
    
    missing_values_list = []
    missing_d = {}
    
    for column in columns:
        missing_count = df[column].isna().sum()
        proportion_missing = missing_count / len(df)
        missing_d[column] = df[column].isna()
        
        # Set the missing value quality based on the proportion of missing values
        if proportion_missing == 0:
            quality = "Excellent (0%)"
        elif proportion_missing < 0.1:
            quality = "Good (<10%)"
        elif proportion_missing < 0.5:
            quality = "Fair (10-50%)"
        elif proportion_missing < 0.9:
            quality = "Poor (50-90%)"
        else:
            quality = "Bad (>90%)"
        
        missing_values_list.append({
            "Column": column,
            "Missing Values": missing_count,
            "Proportion": proportion_missing,
            "Missing Value Quality": quality
        })
    
    # Convert the list of dictionaries to a DataFrame
    missing_values_df = pd.DataFrame(missing_values_list)
    
    # Sort the DataFrame by the proportion of missing values in ascending order
    missing_values_df = missing_values_df.sort_values("Proportion")
    
    return missing_values_df, missing_d




add_stopwords = pd.read_csv(fd + "\\additional files\\stopwords.csv")["stopword"].tolist()
# based on https://gist.github.com/sebleier/554280

def handle_duplicates(df, name_var="name"):
    df = df.reset_index()
    
    # Count the occurrences of each name
    name_counts = df[name_var].value_counts()

    # Filter names that appear more than once
    duplicate_names = name_counts[name_counts > 1].index

    # Create a dictionary to store the renaming mapping
    rename_mapping = {}

    # Rename duplicates with spaces
    for name in duplicate_names:
        indices = df[df[name_var] == name].index
        for i, index in enumerate(indices):
            spaces = ' ' * i
            new_name = spaces + name
            df.at[index, name_var] = new_name
            rename_mapping[name] = new_name

    return df


def choose_first(l, df):
    ll = [c for c in l if c in df.columns]
    if len(ll) > 0:
        return ll[0]
    return None



def split_and_count(value, sep="; "):
    if isinstance(value, str):
        parts = value.split(sep)
        return len(parts)
    else:
        return 0


def freqs_from_list(l, order=1):
    if order == 1:
        fl = [(l.count(x), x) for x in set(l)]
        fl.sort(key=lambda tup: -tup[0])
    elif order == 2:
        fl = [(x, l.count(x)) for x in set(l)]
        fl.sort(key=lambda tup: -tup[1])
    return fl

def freqs_from_ser(s, name=None, rename_dct=None, cumm_perc=True, rel_rank=True):
    f = s.value_counts()
    p = s.value_counts(normalize=True)
    perc = p*100
    df = pd.DataFrame([f,p,perc]).T.reset_index()
    if name is not None:
        df.columns=[name, "f", "f0", "percentage"]
    if cumm_perc:
        df["Cummulative percentage"] = df["f0"].cumsum()
    if rel_rank:
        df["Relative rank"] = df["f"].rank(pct = True) 
    if rename_dct is not None:
        df = df.rename(columns=rename_dct)
    return df

def freqs_from_col(df, name, rename_dct=None):
    if name not in df.columns:
        return None
    return freqs_from_ser(df[name], name=name, rename_dct=rename_dct)

def freqs_to_excel_multiple(df, names, rename_dct=None, f_name="counts.xlsx", index=False):
    writer = pd.ExcelWriter(f_name, engine="xlsxwriter")
    for name in names:
        df0 = freqs_from_col(df, name)
        df0.to_excel(writer, sheet_name=name.replace("\n", " ")[:31], index=index)
    writer.save()      
    
def freqs_of_freqs(counts_df, freq_var="Number of documents", name_var="Name", freq_var_2="Number of",
        split_s="; "):
    grouped_df = counts_df.groupby(freq_var)[name_var].apply(lambda x: split_s.join(x)).reset_index()
    grouped_df.columns = [freq_var, name_var]
    grouped_df[freq_var_2] = grouped_df[name_var].apply(lambda x: len(x.split(split_s)))

    return grouped_df    
    

def dict_from_two_columns(df, x, y):
    return df.set_index(x).to_dict()[y]

def get_indicator_df(df, var, items=None, max_items=None, search="exact", name=None): # , treat_missing_as_0=False
    #dfm = df[var].replace({"nan": np.nan})
    #mask = dfm.isna()
    
    
    df[var] = df[var].astype(str)
    if items is None:
        items = df[var].value_counts().index.astype(str)
    elif len(items) == 0:
        items = df[var].value_counts().index.astype(str)
    if max_items is not None:
        items = items[:max_items]
    if name is None:
        ind_df = pd.DataFrame([], columns=items)
    else:
        ind_df = pd.DataFrame([], columns=[name+"="+it for it in items])
    if search == "exact":
        for it in items:
            v = (df[var]==it).astype(int)
            #if treat_missing_as_0 is False:           
            #    v = np.where(mask, np.nan, v)           
            if name is None:
                ind_df[it] = v
            else:
                ind_df[name + "=" + it] = v
    elif search == "substring":
        for it in items:
            #v = (df[var].astype(str).str.contains(it)).astype(int) # tole iz meni neznanih razlogov ne deluje
            print(it)
            v = [int(it in s) for s in df[var].astype(str)]
            #if treat_missing_as_0 is False:
            #    print(mask)
            #    v = np.where(mask, np.nan, v)    
            if name is None:
                ind_df[it] = v
            else:
                ind_df[name + "=" + it] = v
    return ind_df


def get_indicator_dfs_text(df, var="cleaned abstract", words=[]):

    count_df = pd.DataFrame()

    for word in words:
        count_df[word] = df[var].str.count(word)
    
    presence_df = count_df.copy()
    presence_df[presence_df > 0] = 1
    
    vectorizer = TfidfVectorizer(vocabulary=words)
    tfidf_matrix = vectorizer.fit_transform(df[var])
    
    # Convert the sparse matrix to a dense DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    return count_df, presence_df, tfidf_df


# matrix functions

def pairs_from_sym_matrix(mat, keep_one_pair=False, keep_self=True):
    pairs, items = [], mat.columns # mat mora biti simetrični df, kjer je indeks enak stolpcem  
    for i, it in enumerate(items):
        if keep_one_pair:
            items2 = items[i+2-int(keep_self):]
        else:
            items2 = items
        for it2 in items2:
            if (it==it2) and not keep_self:
                continue
            pairs.append((it, it2, mat.loc[it, it2]))
    pairs_df = pd.DataFrame(pairs, columns=["item 1", "item 2", "value"])    
    return pairs_df.sort_values("value", ascending=False)

def normalize_sq_matrix(mat, method="association"):
    diag = np.diag(mat)
    if method == "association":
        s = mat / np.multiply.outer(diag, diag)
    elif method == "inclusion":
        s = mat / np.minimum.outer(diag, diag)
    elif method == "jaccard":
        s = mat / (np.add.outer(diag, diag) - mat)
    elif method == "salton":
        s = mat / np.sqrt((np.multiply.outer(diag, diag)))
    elif method == "equivalence":
        s = (mat / np.sqrt((np.multiply.outer(diag, diag)))) ** 2
    else:
        print("Unsopported normalization method. No normalization has been made.")
        s = mat
    s[np.isnan(s)] = 0
    return s
    
def normalize_matrix(mat, method="jaccard", check_square=True):
    mat = np.array(mat)
    if check_square:
        if np.array_equal(mat, mat.T):
            return normalize_sq_matrix(mat, method=method)
    if method == "jaccard":
        s = mat / (np.add.outer(mat.sum(axis=1), mat.sum(axis=0))-mat)
    else:
        print("Unsopported normalization method. No normalization has been made.")
        s = mat
    s[np.isnan(s)] = 0
    return s

def net_from_mat(mat, names=None):
    net = nx.from_numpy_array(np.array(mat))
    m = len(net.nodes)
    if names is None:
        names = ["Unit %d" % (i+1) for i in range(m)]
    net_mapping = dict(zip(*(range(len(names)), names)))    
    return nx.relabel_nodes(net, net_mapping)

def get_coocurrence_network(mat=None, ind_mat=None, method="association"):
    if mat is None:
        if ind_mat is None:
            print("No input given. Coocurrence network not computed.")
        else:
            mat = ind_mat.T.dot(ind_mat)
    mat = normalize_sq_matrix(mat, method=method)
    return net_from_mat(mat, names=mat.columns)

def partition_network(net, method=cmt.girvan_newman, **kwds):
    return method(net, **kwds)


def df_from_list_of_dcts(dcts, names=["key", "value"]):
    df = pd.concat([pd.DataFrame.from_dict(d, orient="index") for d in dcts])
    if names is not None:
        df = df.reset_index()
        df.columns = names
    return df

# association between binary variables

def jaccard(a,b,c,d):
    return a/(a+b+c)

def yule_q(a,b,c,d):
    return (a*d-b*c)/(a*d+b*c)

m_fun_dict = {"jaccard": jaccard, "yule_Q": yule_q}

import scipy.stats

def associate_dfs(df1, df2, compute_sig=True, measures=["jaccard", "yule"],
                  correction="hs", alpha=0.05, sort_by="a", ascending=False,
                  remove_empty=True):
    t_names = ["OR", "p-value"] if compute_sig else []
    names = ["group", "x", "a", "b", "c", "d", "gf", "xf", "gf0", "xf0", "cond_g", "cond_x"] + measures + t_names

    measures_fun = [m_fun_dict[m] for m in measures]

    perf = []
    
    if remove_empty:
        zeros_in_df1, zeros_in_df2 = (df1 == 0).all(axis=1), (df2 == 0).all(axis=1)
        common_zeros_indices = zeros_in_df1 | zeros_in_df2
        df1, df2 = df1[~common_zeros_indices], df2[~common_zeros_indices]
    
    df1, df2 = df1.reset_index(drop=True), df2.reset_index(drop=True)
    
    n = len(df1)

    A = df1.T.dot(df2)
    b, c = df1.sum(axis=0), df2.sum(axis=0)
    B = -A.subtract(b, axis="rows")
    C = -A.subtract(c)
    D = n-(A+B+C)

    for x in df2.columns:
        for g in df1. columns:
            a, b, c, d = A.loc[g,x], B.loc[g,x], C.loc[g,x], D.loc[g,x]
            gf, xf = a+b, a+c
            gf0, xf0 = gf/n, xf/n
            cond_g, cond_x = a/(a+b), a/(a+c)
            tmp = [g, x, a, b, c, d, gf, xf, gf0, xf0, cond_g, cond_x]
            for m in measures_fun:
                tmp.append(m(a,b,c,d))
            if compute_sig:
                try:
                    oddsratio, p = scipy.stats.fisher_exact([[a,b],[c,d]])
                    tmp += [oddsratio, p]
                except:
                    tmp += [np.nan, np.nan]
            perf.append(tmp)
           
    perf = pd.DataFrame(perf, columns=names)
    perf = perf.dropna()
    if correction is not None:
        import statsmodels.stats.multitest as multi

        t, corr_p, alpha_sidak, alpha_bonn = multi.multipletests(
            perf["p-value"], alpha=alpha*2, method=correction)

        perf["addjusted p"], perf["sig"] = corr_p, t.astype(int)
    if sort_by is not None:
        perf = perf.sort_values(sort_by, ascending=ascending)

    return perf


# text processing functions
    
def count_un_words(text, sw_l="english", exclude=[], min_len=2):
    tw = [w for w in word_tokenize(text, language=sw_l) if len(w) >= min_len
          if w not in exclude]
    return "; ".join(list(set(tw)))

def count_words_from_text(text):
    while "  " in text:
        text.replace("  ", " ")
    return freqs_from_list(text.split(" "))

from nltk.stem import WordNetLemmatizer


# Define a function to clean and singularize words
def clean_and_singularize_0(word, add_stopwords=add_stopwords):
    # Remove special characters
    word = re.sub("[^A-Za-z]+", "", word)
    # Singularize the word using the WordNetLemmatizer from nltk
    lemmatizer = WordNetLemmatizer()
    word = lemmatizer.lemmatize(word, pos="n")
    if word in add_stopwords:
        return ""
    return word

def clean_and_singularize(word, add_stopwords=add_stopwords):
    if " " in word:
        return " ".join([clean_and_singularize_0(w, add_stopwords=add_stopwords) for w in word.split()])
    return clean_and_singularize_0(word, add_stopwords=add_stopwords)
    

def lemmatize_text(text, sw_l="english", add_stopwords=add_stopwords, lf=lf, exclude=[], min_len=2):

    if (text != text) or (text is None):
        return ""
    text = text.lower()
    if sw_l is not None:
        exclude += set(stopwords.words(sw_l)+add_stopwords)
    
    tw = [w for w in word_tokenize(text, language=sw_l) if len(w) >= min_len
              if w not in exclude]

    if lf is not None:
        tw = [lf(w) for w in tw]
    return " ".join(tw)


def _get_ngrams(t, n):
    n_grams = ngrams(t.split(), n)
    return [" ".join(grams) for grams in n_grams]

def get_ngrams(tw, max_n=1):
    if max_n == 1:
        return tw
    return unnest([_get_ngrams(tw, n) for n in range(2, max_n+1)])


def count_ngrams_from_text(text, max_n=2):
    ng = get_ngrams(text, max_n=max_n)
    return freqs_from_list(ng)

def count_words_and_phrases(text, max_n=2):
    f1, f2 = count_words_from_text(text), count_ngrams_from_text(text, max_n=max_n)
    return sorted(f1+f2, reverse=True)
    
def get_tfidf_features(s, ng=2, lang="english", add_stopwords=add_stopwords,
                       min_word_len=3, exclude_numbers=True,
                       max_df=1., min_df=1, max_words=30, 
                       norm="l2", sublinear_tf=True, **kwds):
    s = s.replace({np.nan: "", None: ""})
    stop_words = list(stopwords.words(lang)) + add_stopwords
    if exclude_numbers:
        stop_words = [w for w in stop_words if not is_float(w)]
    tfidf = TfidfVectorizer(encoding="utf-8", ngram_range=(1,ng),
                            stop_words=stop_words,
                            lowercase=True, max_df=max_df, min_df=min_df,
                            max_features=max_words, norm=norm, 
                            sublinear_tf=sublinear_tf, **kwds)
    features = tfidf.fit_transform(s).toarray()
    df = pd.DataFrame(features, columns=tfidf.get_feature_names())
    keep = [x for x in df.columns 
            if len(x) >= min_word_len if x not in add_stopwords]
    if exclude_numbers:
        keep = [x for x in keep if not is_float(x)]
    return df[keep]

# structured text (for example "keywords") preprocessing functions

def replacer_df_bibliometrix_T(df):
    # ime za nasometstilo - ime stolpca, nadomeščene vrednosti - v stolpcu
    # po zgledu bibliometrix replacerjev, le da je v oblki df in transponiran
    d = {}
    for c in df.columns:
        for v in df[c]:
            d[v] = c
    return d
    
def lemmatize_keywords(text, sw_l="english", lf=lf, s_char="; "): 
    if (text != text) or (text is None):
        return ""
    text = text.lower()
    kws = text.split(s_char)
    lkws = []
    for kw in kws:
       lkws.append(" ".join([lf(w) for w in word_tokenize(kw, language=sw_l)]))
    return s_char.join(lkws)

def replace_keywords(text, replacer_d={}, s_char="; "):
    def rf(s): return replacer_d[s] if s in replacer_d else s
    if text != text:
        return ""
    if text is None:
        return ""
    return s_char.join([rf(w) for w in text.split(s_char)])

def remove_keywords(text, remove=[], s_char="; "):
    w = text.split(s_char)
    for r in remove:
        if r in w:
            w.remove(r)
    return s_char.join(w)

def preprocess_keywords(text, sw_l="english", lf=lf, s_char="; ", replacer_d={}, remove=[]): # lemmatizer=WordNetLemmatizer(),
    if type(replacer_d) == pd.core.frame.DataFrame:
        replacer_d = replacer_d.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        replacer_d = replacer_d.set_index(replacer_d.columns[0]).to_dict()[replacer_d.columns[1]]
    
    if lf is not None:
        text = lemmatize_keywords(text, sw_l=sw_l, lf=lf, s_char=s_char)
    if len(replacer_d) > 0:
        text = replace_keywords(text, replacer_d=replacer_d, s_char=s_char)
    if len(remove) > 0:
        text = remove_keywords(text, remove=remove, s_char=s_char)
    if text != text:
        return ""
    return text

def preprocess_text(text, replacer_d={}, remove=[]): # dodaj še opcijo za remove, da je lahko dataframe
    
    if type(replacer_d) == pd.core.frame.DataFrame:
        replacer_d = replacer_d.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        replacer_d = replacer_d.set_index(replacer_d.columns[0]).to_dict()[replacer_d.columns[1]]
    
    for key, value in replacer_d.items():
        # Use word boundaries to ensure whole-word replacement
        text = re.sub(rf'\b{re.escape(key)}\b', value, text)

    # Remove words based on the remove list
    for word in remove:
        # Use word boundaries to ensure whole-word removal
        text = re.sub(rf'\b{re.escape(word)}\b', '', text)

    # Optionally, strip extra spaces that might result from the replacements or removals
    text = ' '.join(text.split())

    return text
    




def preprocess_keywords_df(df, which="aik", lf=None, replacer_d={}, remove=[], lang_docs="en", drop_capital=True): #
    if which in ["ak", "aik"]:
        kw_var = first_element(["author keywords", "authors keywords", "Author Keywords", "Authors Keywords"], df.columns)
        df[kw_var] = df[kw_var].str.lower()
        df["author keywords"] = df[kw_var].map(
            lambda x: preprocess_keywords(
                x, sw_l=lang_docs, lf=lf, 
                s_char="; ", replacer_d=replacer_d, remove=remove))
    if which in ["ik", "aik"]:
        kw_var = first_element(["indexed keywords", "index keywords", "Indexed Keywords", "Index Keywords"], df.columns)
        df[kw_var] = df[kw_var].str.lower()
        df["indexed keywords"] = df[kw_var].map(
            lambda x: preprocess_keywords(
                x, lf=lf, sw_l=lang_docs, 
                s_char="; ", replacer_d=replacer_d, remove=remove))
    if drop_capital:
        drop = [c for c in df.columns if c in ["Author Keywords", "Indexed Keywords"]]
        df = df.drop(columns=drop)
    return df



# functions for computation of new variables based on condidions and one variable

def compute_variable(df, v, words, name="pillar", search="substring", cond="|"):
    if search == "substring":
        if cond in ["|", "or"]:
            cond = cond.join(words)
        elif cond in ["and", "&", "(="]:
            words = ["(?=.*"+w for w in words]
            cond = ")".join(words) + ")"
        df[name] = (df[v].astype(str).str.contains(cond)).astype(int)
    elif search == "exact":
        df[name] = (df[v].isin(words)).astype(int)
    return df

def compute_variables(df, v, l_words, names=[], search="substring", cond="|"):
    k = len(l_words)
    if type(search) == str:
        search = [search] * k
    if type(cond) == str:
        cond = [cond] * k
    if len(names) == 0:
        names = ["pillar %d" % (i+1) for i in range(k)]
    for i in range(k):
        df = compute_variable(df, v, l_words[i], name=names[i], search=search[i], cond=cond[i])
    return df

def link_terms(df_terms):
    # prvi stolpec pojmi, drugi navezave za pojme
    d_terms = {}
    for term in df_terms[df_terms.columns[0]].unique():
        d_terms[term] = df_terms[df_terms[df_terms.columns[0]]==term][df_terms.columns[1]].tolist()
    return d_terms
        
def compute_variables_list_terms(df, v, df_terms, search="substring", cond="|"):
    d_terms = link_terms(df_terms)
    return compute_variables(df, v, list(d_terms.values()), names=list(d_terms.keys()), search=search, cond=cond)



# 




def warn(obj, att, fn=None, kind=1): # dodaj še za stolpce v dataframe-u
    if not hasattr(obj, att) and kind ==1:
        print(f"{obj} has not attribute {att}. Call {fn} to compute {att}.")
        return None
    if att not in obj.columns and kind == 2:
        print(f"Dataframe {obj} has no column {att}")
        return None


# clustering

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# optimal k for k-means
def score_partitions(df, method="davies_bouldin", min_k=2, max_k=20, label=True):
    if method in ["davies_bouldin"]:
        reverse=True
    scores = []
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k)
        model = kmeans.fit_predict(df)
        if label:
            df["cluster members %d" %k] = model 
        if method == "davies_bouldin":
            score = davies_bouldin_score(df, model)
            scores.append([score, k, model])
        else:
            return None
    opt = sorted(scores, reverse=reverse)[0]
    opt_k = opt[1]
    if label:
        df["cluster members optimal %d" %opt_k] = df["cluster members %d" %opt_k]
    return df, opt


def get_opt_k(df, method="davies_bouldin", min_k=2, max_k=20):
    df, part = score_partitions(df, method=method, min_k=min_k, max_k=max_k, label=False)
    return part[0]


# MDS, network from distances 

from scipy.spatial import distance
from sklearn.manifold import MDS, TSNE

def distances_from_df(df, what="variables", metric="cosine"):
    if what == "variables":
        df = df.T    
    dd = distance.pdist(df, metric=metric)
    return distance.squareform(dd)

def mds_from_df(df, nc=2, what="variables", metric="cosine"):
    embedding = MDS(n_components=nc, dissimilarity="precomputed")
    d = distances_from_df(df, what=what, metric=metric)
    return pd.DataFrame(embedding.fit_transform(d), columns=["MDS %d" % i for i in range(nc)])

def tsne_from_df(df, nc=2, what="variables", method="barnes_hut", **kwds):
    embedding = TSNE(n_components=nc, method=method, **kwds)
    if what == "variables":
        df = df.T
    return pd.DataFrame(embedding.fit_transform(df), columns=["t-SNE %d" % i for i in range(nc)])



from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def compute_dissimilarity_matrix(df, metric='jaccard'):
    """
    Compute the dissimilarity matrix for a given binary dataframe.
    
    Parameters:
    df (pd.DataFrame): Binary dataframe with first column as names.
    metric (str): Dissimilarity metric. Options: 'jaccard', 'hamming', 'euclidean'.
    
    Returns:
    np.ndarray: Dissimilarity matrix.
    """
    data = df.iloc[:, 1:].values
    dist_matrix = pdist(data, metric=metric)
    return dist_matrix

def perform_hierarchical_clustering(dist_matrix, method='ward'):
    """
    Perform hierarchical clustering.
    
    Parameters:
    dist_matrix (np.ndarray): Dissimilarity matrix.
    method (str): Linkage method. Options: 'single', 'complete', 'average', 'ward'.
    
    Returns:
    np.ndarray: Linkage matrix.
    """
    linkage_matrix = linkage(dist_matrix, method=method)
    return linkage_matrix


# text functions

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

add_stopwords = pd.read_csv(fd + "\\additional files\\stopwords.csv")["stopword"].tolist()

def text_preprocess(text, exclude=[], remove_numbers=True):
    if text is None or pd.isna(text):
        return text
   
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in add_stopwords if word not in exclude]
    
    if remove_numbers:
        tokens = [word for word in tokens if not word.isdigit()]
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def count_ngrams(df, text_column, n):
    
    ndoc = len(df)
    
    # Initialize CountVectorizer for word counts
    word_vectorizer = CountVectorizer()
    word_counts = word_vectorizer.fit_transform(df[text_column])
    
    # Get word counts as a DataFrame
    word_counts_df = pd.DataFrame(word_counts.toarray(), columns=word_vectorizer.get_feature_names_out())
    
    # Total word counts
    total_word_counts = word_counts_df.sum().to_dict()
    
    # Rows where each word appears at least once
    word_in_row_counts = (word_counts_df > 0).sum().to_dict()
    
    # Initialize CountVectorizer for n-grams
    if n > 1:
        ngram_vectorizer = CountVectorizer(ngram_range=(2, n))
        ngram_counts = ngram_vectorizer.fit_transform(df[text_column])
    
        # Get n-gram counts as a DataFrame
        ngram_counts_df = pd.DataFrame(ngram_counts.toarray(), columns=ngram_vectorizer.get_feature_names_out())
    
        # Total n-gram counts
        total_ngram_counts = ngram_counts_df.sum().to_dict()
    
        # Rows where each n-gram appears at least once
        ngram_in_row_counts = (ngram_counts_df > 0).sum().to_dict()
    else:
        ngram_in_row_counts, total_ngram_counts = {}, {}
    
    # Merge the dictionaries
    total_counts = {**total_word_counts, **total_ngram_counts}
    row_counts = {**word_in_row_counts, **ngram_in_row_counts}
    
    # Sort by frequency in the total_counts dictionary
    sorted_total_counts = dict(sorted(total_counts.items(), key=lambda item: item[1], reverse=True))
    
    # Convert to DataFrame
    counts_df = pd.DataFrame(sorted_total_counts.items(), columns=["word/phrase", "total count"])
    
    # Add row_count to the DataFrame
    counts_df["row count"] = counts_df["word/phrase"].map(row_counts)
    
    counts_df["row count relative"], counts_df["total count relative"] = counts_df["row count"]/ndoc, counts_df["total count"]/ndoc
    
    return counts_df



# topic modelling

import gensim
import pprint

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts, stop_words=add_stopwords):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def lda_topics(df, text_var, model=gensim.models.LdaModel, num_topics=10, stop_words=add_stopwords):
    # opcija za model gensim.models.LdaMulticore -> za ostalo preveri
    data = df[text_var].values.tolist()
    data_words = list(sent_to_words(data))
    data_words = remove_stopwords(data_words,  stop_words=stop_words)
    id2word = gensim.corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = model(corpus=corpus, id2word=id2word, num_topics=num_topics)

    #pprint.pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    return lda_model, doc_lda, id2word, corpus 

# relation between wto concepts

from sklearn.cross_decomposition import CCA, PLSCanonical, PLSSVD, PLSRegression

def concat_projections(X, Y):
    n, nc = X.shape
    x = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(nc)])
    y = pd.DataFrame(Y, columns=[f"Y{i+1}" for i in range(nc)])
    return pd.concat([x, y], axis=1)

def get_loadings_dfs(X, Y, x_names=None, y_names=None):
    nc = X.shape[1]
    x_cols, y_cols = [f"X{i+1}" for i in range(nc)], [f"Y{i+1}" for i in range(nc)] 
    return pd.DataFrame(X, columns=x_cols, index=x_names), pd.DataFrame(Y, columns=y_cols, index=y_names)

class TwoConcepts:
    
    def __init__(self, df_i1, df_i2):
        self.df_i1, self.df_i2 = df_i1, df_i2
        self.x_names, self.y_names = df_i1.columns, df_i2.columns
        self.d12 = df_i1.T.dot(df_i2)
        self.d21 = df_i2.T.dot(df_i1)
        
        self.d12_ind = self.d12.applymap(lambda x: 1 if x != 0 else 0)
        self.d21_ind = self.d21.applymap(lambda x: 1 if x != 0 else 0)
                
    def cca(self, nc=2):
        self.cca = CCA(n_components=nc).fit(self.df_i1, self.df_i2)
        self.cca_vars_X, self.cca_vars_Y = self.cca.transform(self.df_i1, self.df_i2)
        self.cca_proj_df = concat_projections(self.cca_vars_X, self.cca_vars_Y)   
        self.cca_loadings_X, self.cca_loadings_Y = self.cca.x_loadings_, self.cca.y_loadings_
        self.cca_loadings_X_df, self.cca_loadings_Y_df =\
            get_loadings_dfs(self.cca_loadings_X, self.cca_loadings_Y, x_names=self.x_names, y_names=self.y_names)
        
    def plsca(self, nc=2):
        self.plsca = PLSCanonical(n_components=nc).fit(self.df_i1, self.df_i2)
        self.plsca_vars_X, self.plsca_vars_Y = self.plsca.transform(self.df_i1, self.df_i2)
        self.plsca_proj_df = concat_projections(self.plsca_vars_X, self.plsca_vars_Y)   
        self.plsca_loadings_X, self.plsca_loadings_Y = self.plsca.x_loadings_, self.plsca.y_loadings_
        self.plsca_loadings_X_df, self.plsca_loadings_Y_df =\
            get_loadings_dfs(self.plsca_loadings_X, self.plsca_loadings_Y, x_names=self.x_names, y_names=self.y_names)
        
    def plssvd(self, nc=2):
        self.plssvd = PLSSVD(n_components=nc).fit(self.df_i1, self.df_i2)
        self.plssvd_vars_X, self.plssvd_vars_Y = self.plssvd.transform(self.df_i1, self.df_i2)
        self.plssvd_proj_df = concat_projections(self.plssvd_vars_X, self.plssvd_vars_Y)   
        self.plssvd_loadings_X, self.plssvd_loadings_Y = self.plssvd.x_weights_, self.plssvd.y_weights_
        self.plssvd_loadings_X_df, self.plssvd_loadings_Y_df =\
            get_loadings_dfs(self.plssvd_loadings_X, self.plssvd_loadings_Y, x_names=self.x_names, y_names=self.y_names)
            
    def plsreg(self, nc=2):
        self.plsreg = PLSRegression(n_components=nc).fit(self.df_i1, self.df_i2)
        self.plsreg_vars_X, self.plsreg_vars_Y = self.plsreg.transform(self.df_i1, self.df_i2)
        self.plsreg_proj_df = concat_projections(self.plsreg_vars_X, self.plsreg_vars_Y)  
        self.plsreg_loadings_X, self.plsreg_loadings_Y = self.plsca_loadings_X, self.plsca_loadings_Y
        self.plsreg_loadings_X_df, self.plsreg_loadings_Y_df =\
            get_loadings_dfs(self.plsreg_loadings_X, self.plsreg_loadings_Y, x_names=self.x_names, y_names=self.y_names)
            
# multiple concepts

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


# comparison of proportions

def get_props_df(Props, props):
    props_df = pd.concat([Props, props], axis=1, sort=True)
    props_df.columns = ["Prop", "prop"]
    props_df["Prop %"], props_df["prop %"] = props_df["Prop"] *100, props_df["prop"] *100   
    props_df["difference"] = props_df["prop"] - props_df["Prop"]
    props_df["relative difference"] = props_df["prop"] / props_df["Prop"] - 1
    props_df["PP difference"] = props_df["difference"] *100
    props_df["P difference"] = props_df["relative difference"] *100
    return props_df


# Lotka's law
    
def compute_lotka_law(n_values, f_values, verbose=2):
    # Define the Lotka's Law function
    def lotka_law(n, C, n_param):
        return C / (n ** n_param)

    # Fit the model
    params, covariance = curve_fit(lotka_law, n_values, f_values, p0=[1, 2])

    # Extract the parameters
    C, n_param = params

    # Calculate predicted values
    predicted_values = lotka_law(n_values, C, n_param)

    # Calculate residuals
    residuals = f_values - predicted_values

    # R-squared
    r_squared = r2_score(f_values, predicted_values)

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(f_values, predicted_values))

    # Chi-squared test
    chi_squared = np.sum((residuals**2) / predicted_values)

    # KS Test
    ks_stat, p_value = ks_2samp(f_values, predicted_values)
    
    if verbose > 2:
        print(f"Estimated C: {C}")
        print(f"Estimated n: {n_param}")
        print(f"R-squared: {r_squared}")
        print(f"RMSE: {rmse}")
        print(f"Chi-squared: {chi_squared}")
        print(f"KS Statistic: {ks_stat}")
        print(f"P-value: {p_value}")

    results = {
        'C': C,
        'n_param': n_param,
        'r_squared': r_squared,
        'rmse': rmse,
        'chi_squared': chi_squared,
        'ks_stat': ks_stat,
        'p_value': p_value,
        'predicted_values': predicted_values,
        'residuals': residuals,
    }

    return results