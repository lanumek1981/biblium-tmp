# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:49:18 2022

@author: Lan
"""

import pandas as pd
import numpy as np
import os
import re

fd = os.path.dirname(__file__)
df_d = pd.read_excel(fd + "\\additional files\\databases field names.xlsx",
                     sheet_name="variable names")
df_d = df_d.rename(columns=lambda x: x.lower())

core_vars = df_d[df_d["core"]==1]["name"].tolist()

# misc function for WoS

def nam_val(s):
    return (s.split(" ")[0], " ".join(s.split(" ")[1:]))

def record_to_df(rec):
    rr = rec.replace("\n   ", "\t").split("\n")
    df = pd.DataFrame([nam_val(r) for r in rr[:-1]]).drop_duplicates(subset=[0])
    return df

def records_to_df(records):
    dfs = [record_to_df(rec).set_index(0) for rec in records]
    return pd.concat(dfs, axis=1).T

def merge_wos_files(files, save_name="wos.txt"):
    outs = []
    for file in files:
        f = open(file, "rt", encoding="UTF-8-sig")
        cnt = f.read()
        f.close()
        head = cnt[:45]
        outs.append(cnt[45:-2])
    f_out = open(save_name, "wt", encoding="UTF-8-sig")
    f_out.write(head + "\n".join(outs) + "EF")
    f_out.close()
    

d_wos = df_d.set_index("wos").to_dict()["name"]
    
def read_wos_txt(file, clean=True, d=d_wos, keep_just_core=True):
    f = open(file, "rt", encoding="UTF-8-sig", errors="ignore")
    cnt = f.read()[45:]

    f.close()    
    c = cnt.split("ER\n\n")
    df_out = records_to_df(c[:-1])
    if clean:
        g1 = ["AU", "AF", "BE"]
        g2 = ["TI", "SO", "DE", "ID", "EM", "RI", "OI", "RI", "OI", "FU", "FX", "PU", "PI", "PA", "SN", "EI", "J9", "JI",  "WC", "SC", "GA"]
        g3 = ["C1", "CR"]
        g4 = ["NR", "TC", "Z9", "U1", "U2", "PY"]
        for c in df_out.columns:
            if c in g1:
                df_out[c] = df_out[c].astype(str).str.replace("\t", "; ")
            if c in g2:
                df_out[c] = df_out[c].astype(str).str.replace("\t", " ")
            if c in g3:
                df_out[c] = df_out[c].astype(str).str.replace(";", "")
                df_out[c] = df_out[c].astype(str).str.replace("\t", "; ")
            if c in g4:
                df_out[c] = df_out[c].replace({"nan": 0})
                df_out[c] = df_out[c].astype(np.float).astype("Int32")
            if c not in g4:
                df_out[c] = df_out[c].fillna("nan")
                df_out[c] = df_out[c].replace({"nan": np.nan})
    df_out = df_out.replace({pd.NaT: np.nan})
    df_out = df_out.rename(columns=d)    
    if keep_just_core:
        df_out = df_out[[c for c in df_out.columns if c in core_vars]]
    return df_out

def read_wos_xls(f_name):
    pass


# scopus

def merge_scopus_files(files):
    dfs = []
    for f_name in files:
        print(f_name)
        if ".xlsx" in f_name:
            df = pd.read_excel(f_name, on_bad_lines="skip")
        elif ".csv" in f_name:
            df = pd.read_csv(f_name, on_bad_lines="skip")
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def read_bibfile(f_name, database):
    if f_name is None:
        return pd.DataFrame([])
    if database.lower() == "scopus":
        if ".xlsx" in f_name:
            df = pd.read_excel(f_name)
        elif ".csv" in f_name:
            df = pd.read_csv(f_name)
    elif database.lower() == "wos":
        if ".xls" in f_name:
            df = read_wos_xls(f_name)
        elif ".txt" in f_name:
            df = read_wos_txt(f_name)
    return df



# special function for reading SDG data from scopus

def format_sdg(sdg_str):
    pattern = re.compile(r"SDG ([1-9])\b")
    result = re.sub(pattern, lambda match: f"SDG {match.group(1).zfill(2)}", sdg_str)
    return result

def sdg_files_to_df(ld=None, fd=None):
    
    if ld is None:
        if fd is not None:
            ld = os.listdir(fd)
        else:
            print("No input given.")
            return None

    dfs = []
    
    for f in ld:
        df_tmp = pd.read_csv(os.getcwd()+"\\data\\partial\\" + f)
        df_tmp["SDG " + f.split("(")[1].split(")")[0]] = 1
        dfs.append(df_tmp)
        
    merged_df = pd.concat(dfs, sort=False)
    
    d1 = merged_df[["EID"] + [c for c in merged_df.columns if "SDG" in c]]
    d1 = d1.fillna(0)
    d1 = d1.groupby("EID").agg(max).reset_index()
    
    d2 = merged_df[[c for c in merged_df.columns if "SDG" not in c]]
    d2 = d2.drop_duplicates(subset=["EID"])
    
    df = pd.merge(d1, d2, on="EID")
    
    df = df.rename(columns = lambda x: format_sdg(x))
    
    df["life"] = df[["SDG 01", "SDG 02", "SDG 03"]].max(axis=1)
    df["economic and technological development"] = df[["SDG 08", "SDG 09"]].max(axis=1)
    
    df["social development"] = df[["SDG 11", "SDG 16"]].max(axis=1)
    df["equality"] = df[["SDG 04", "SDG 05", "SDG 10"]].max(axis=1)
    
    df["resources"] = df[["SDG 06", "SDG 07", "SDG 12"]].max(axis=1)
    df["natural environment"] = df[["SDG 13", "SDG 14", "SDG 15"]].max(axis=1)
    
    df["economic dimension"] = df[["life", "economic and technological development"]].max(axis=1)
    df["social dimension"] = df[["social development", "equality"]].max(axis=1)
    df["environmental dimension"] = df[["resources", "natural environment"]].max(axis=1)
    
    return df