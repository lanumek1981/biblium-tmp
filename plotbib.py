# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:48:14 2022

@author: lan
"""

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import miscbib, utilsbib
import venn
import datetime
from matplotlib.colors import Normalize
from collections import defaultdict
import matplotlib

from textwrap import fill, wrap

from adjustText import adjust_text
#from matplotlib.colors import Normalize

import networkx as nx
#import plotly.graph_objects as go

# general

def barh_plot_df(df, x_var, y_var, c_var=None, cmap="plasma", fontsize=7, fill_width=60, 
                f_name=None, dpi=1200, n_digit=3, **kwds):
    fig = plt.figure()
    df = df.sort_values(y_var, ascending=True)
    
    scalarmappaple = cm.ScalarMappable(cmap=cmap)
    
    if c_var is not None:
        c = df[c_var]
        scalarmappaple.set_array(c)
        ax = df.plot.barh(x_var, y_var, color=scalarmappaple.to_rgba(c), **kwds)  
    else:
        ax = df.plot.barh(x_var, y_var, **kwds)
          
    ax.set_yticklabels(fill(y.get_text(), fill_width) 
                       for y in ax.get_yticklabels())
    for p in ax.patches:
        ax.text(p.get_width() + 0.1, p.get_y()+0.1, 
                str(np.round(p.get_width(), decimals=n_digit)),
                fontsize=fontsize)
    ax.set_title(y_var)
    ax.get_legend().remove()
    
    if c_var is not None:
        c = df[c_var]
        scalarmappaple.set_array(c)
        cbar = plt.colorbar(scalarmappaple, label=c_var)
        cbar.set_ticks([c.min(), c.max()])
        if pd.api.types.is_numeric_dtype(c):
            if c.max() - c.min() >= 1:
                cbar.set_ticklabels([int(np.ceil(c.min())), int(np.floor(c.max()))])
            else:
                cbar.set_ticklabels([np.round(c.min(), n_digit), np.round(c.max(), n_digit)])
    
    plt.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")


def lollipop_plot_df(df, x_var, y_var, c_var=None, cmap="plasma", def_color="blue", fontsize=7, fill_width=60, 
                f_name=None, dpi=1200, n_digit=3, max_size = 20, **kwds):

    fig, ax = plt.subplots()

    df0 = df.sort_values(x_var, ascending=False).reset_index()
    
    x = df0[x_var].iloc[::-1]
    if x_var in ["Number of documents", "Total number of citations", "H-index", "G-index"]:
        x = x.astype(int)
    else:
        x = np.round(x, n_digit)
    n = len(x)
    my_range=range(1,n+1)

    if c_var is not None:
        c = df0[c_var].iloc[::-1]
        scalarmappaple = cm.ScalarMappable(cmap=cmap)
        scalarmappaple.set_array(c)
        colors = scalarmappaple.to_rgba(c)
    else:
        colors = [def_color] * len(df0)
    
    plt.hlines(y=my_range, xmin=[0]*n, xmax=x, color=colors[::-1])
    for i in range(n):
        plt.plot(x[i], n-i, "o", markersize=x[i]/x[0]*max_size, color=colors[i])
        plt.text(x[i]*1.05, n-i, str(np.round(x[i],n_digit)))
    
    ax.set_xlim((0, max(x)*1.1))
    plt.yticks(my_range, df0[y_var].iloc[::-1])
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    
    if c_var is not None:
        if c.max() - c.min() >= 1:
            plt.colorbar(scalarmappaple, label=c_var,
                         ticks=(np.ceil(c.min()), np.floor(c.max())))           
        else:
            plt.colorbar(scalarmappaple, label=c_var,
                         ticks=(np.round(c.min(), 1),
                                np.round(c.max(), 1)))
    
    plt.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")


def heatmap_df(df, cmap="viridis", annot=True, fmt=".2g", label_cbar=None, 
               fmt_cbar="%.2g", xticklabels=True, yticklabels=True,
               xlabel=None, ylabel=None, f_xlabel=10, f_ylabel=10,
               f_colorbar=10, f_lab_colorbar=10,
               annot_size=10, ylabel_size=10, xlabel_size=10,
               prepocess_columns=False, title=None, f_name=None, dpi=1200, **kwds):
    if prepocess_columns:
        if " =" in df.columns[0]:
            label = df.columns[0].split(" =")[0]
            if xlabel is None:
                xlabel = label
        df = df.rename(columns=lambda x: x.split(" = ")[1])
    n = df.shape[0]
    f, ax = plt.subplots()
    if annot_size is None: annot_size = int(200/n)
    if ylabel_size is None: ylabel_size = int(250/n)
    if xlabel_size is None: xlabel_size = ylabel_size
    if f_xlabel is None: f_xlabel = xlabel_size
    if f_ylabel is None: f_ylabel = ylabel_size
    m, M = df.to_numpy().min(), df.to_numpy().max()
    ax = sns.heatmap(df, cmap=cmap, annot=annot, annot_kws={"size": annot_size}, 
                 cbar_kws={"label": label_cbar, "ticks": [m, M], "format": fmt_cbar}, 
                 xticklabels=xticklabels, yticklabels=yticklabels,
                 fmt=fmt, **kwds)   

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=f_colorbar)
    cbar.set_label(label=label_cbar, size=f_lab_colorbar)

    plt.yticks(size=ylabel_size)
    plt.xticks(size=xlabel_size)

    plt.xlabel(xlabel, fontsize=f_xlabel)
    plt.ylabel(ylabel, fontsize=f_ylabel)
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")  
        plt.savefig(f_name + ".pdf")        
        
        
def scatter_plot_df(df, x_var, y_var, 
                    s_var=None, c_var=None, l_var=None, 
                    max_size=100, show_legend=True,
                    x_scale="log", y_scale="log",
                    cmap="plasma", d_cmap="Set1",
                    f_name="scatter", 
                    dpi=1200, digits=2, core=True,
                    save_results=True,
                    show_mean_line=False, show_mu=False,
                    show_median_line=False, show_mid_line=False,
                    show_reg_line=False, show_id_line=False,
                    show_k_lines=[], show_int_lines=[], sym_lines=True,
                    show_axis=False,
                    mu_size=6, label_fontsize=6, leg_tit_fontsize=8,
                    thr_text=None, max_text_len=None,
                    leg_fontsize=6, leg_size=6,
                    connect_pairs=None, thr_connect=0.5,
                    arrowprops=None, 
                    loc_legend_color=(1.01,0), loc_legend_size=4,
                    n_dots_legend=5,
                    axis_off=False, keep_just_frame=False,
                    remove_axis=False,
                    x_label=None, y_label=None,
                    remove_ticks=False,
                    **kwds):
    #df = df.dropna()
    x, y = df[x_var], df[y_var]
    
    fig = plt.figure()
    ax = plt.gca()
    
    s, smax = None, 100
    if s_var is not None:
        smax = df[s_var].max()
        s = max_size * df[s_var] / smax
        
    c = None
    scalarmappaple = cm.ScalarMappable(cmap=cmap)
    if c_var is not None:
        c = df[c_var]
        if pd.api.types.is_numeric_dtype(c):
            
            scalarmappaple.set_array(c)
            cbar = plt.colorbar(scalarmappaple, label=c_var, ax=ax)
            cbar.set_ticks([c.min(), c.max()])
            if c.max() - c.min() >= 1:
                cbar.set_ticklabels([int(np.ceil(c.min())), int(np.floor(c.max()))])
            else:
                cbar.set_ticklabels([np.round(c.min(), digits), np.round(c.max(), digits)])
        elif pd.api.types.is_string_dtype(c):
            scalarmappaple = cm.ScalarMappable(cmap=d_cmap)
            cmap = d_cmap
            vals = df[c_var].unique()
            k = len(vals)
            cd = dict(zip(*(vals, scalarmappaple.cmap(np.linspace(0,1,k)))))
            c = [cd[v] for v in list(df[c_var])]
            cu = [cd[v] for v in df[c_var].unique()]

    scatter = plt.scatter(x, y, s=s, c=c, cmap=cmap, **kwds)
    
    if show_mean_line:
        mu_x, min_x, max_x = df[x_var].mean(), df[x_var].min(), df[x_var].max()
        mu_y, min_y, max_y = df[y_var].mean(), df[y_var].min(), df[y_var].max()
    
        ax.plot([mu_x,mu_x], [min_y, max_y], "k--", linewidth=1)
        ax.plot([min_x, max_x], [mu_y,mu_y], "k--", linewidth=1)
        
        if show_mu:
            ax.text(mu_x * 0.95, min_y * 0.95, 
                    r"$\mu _x=%3.2f$" % mu_x, fontsize=mu_size)       
            ax.text(min_x, mu_y * 1.05, 
                    r"$\mu _y=%3.2f$" % mu_y, fontsize=mu_size)
    
    if show_median_line:
        me_x, min_x, max_x = df[x_var].median(), df[x_var].min(), df[x_var].max()
        me_y, min_y, max_y = df[y_var].median(), df[y_var].min(), df[y_var].max()
    
        ax.plot([me_x,me_x], [min_y, max_y], "k--", linewidth=1)
        ax.plot([min_x, max_x], [me_y,me_y], "k--", linewidth=1)
        
    if show_mid_line:
        min_x, max_x = df[x_var].min(), df[x_var].max()
        min_y, max_y = df[y_var].min(), df[y_var].max()
        
        mid_x, mid_y = 0.5*(min_x+max_x), 0.5*(min_y+max_y)
    
        ax.plot([mid_x,mid_x], [min_y, max_y], "k--", linewidth=1)
        ax.plot([min_x, max_x], [mid_y,mid_y], "k--", linewidth=1)
        
            
    if show_axis:
        min_x, max_x = df[x_var].min(), df[x_var].max()
        min_y, max_y = df[y_var].min(), df[y_var].max()
        
        ax.plot([0,0], [min_y, max_y], "k", linewidth=1)
        ax.plot([min_x, max_x], [0,0], "k", linewidth=1)
        
    if show_reg_line:
        
        idx = np.isfinite(df[x_var]) & np.isfinite(df[y_var])
        m, b = np.polyfit(df[x_var][idx], df[y_var][idx], 1)
        
        plt.plot(df[x_var], m*df[x_var]+b, "k--", linewidth=1)
        
    if show_id_line:
        plt.plot(df[x_var].sort_values(), df[x_var].sort_values(), "k--", linewidth=1)
        
    if len(show_k_lines):
        x_ = df[x_var].sort_values()
        for k in show_k_lines:
            plt.plot(x_, k*x_, "k--", linewidth=0.5)
            if sym_lines:
                plt.plot(x_, x_/k, "k--", linewidth=0.5)
            
    if len(show_int_lines):
        x_ = df[x_var].sort_values()
        for k in show_int_lines:
            plt.plot(x_, x_+k, "k--", linewidth=0.5)
            if sym_lines:
                plt.plot(x_, x_-k, "k--", linewidth=0.5)
        
        
    ax.set_yscale(y_scale); ax.set_xscale(x_scale)
    if x_label is None:
        x_label = x_var
    if y_label is None:
        y_label = y_var
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    
    if l_var is not None:
        if max_text_len is not None:
            df[l_var] = df[l_var].apply(lambda x: x[:max_text_len])
        
        texts = []
        for index, row in df.iterrows():
            if thr_text is None:
                texts.append(plt.text(
                    row[x_var], row[y_var], row[l_var], 
                    fontsize=label_fontsize))
            else:
                if max(abs(row[x_var]), abs(row[y_var])) >= thr_text:
                    texts.append(plt.text(
                        row[x_var], row[y_var], row[l_var], 
                    fontsize=label_fontsize))
            
        if arrowprops is not None:
            adjust_text(texts, arrowprops=arrowprops) 
            # For example: arrowprops=dict(arrowstyle="fancy",
            # color="k", alpha=.5)
        else:
            adjust_text(texts)
    
    if connect_pairs is not None:
        for i1, r1 in df.iterrows():
            for i2, r2 in df.iterrows():
                linewidth = connect_pairs[r1[l_var]][r2[l_var]]
                if linewidth < thr_connect:
                    linewidth = 0
                plt.plot([r1[x_var], r2[x_var]], [r1[y_var], r2[y_var]],
                         color="k", linewidth=linewidth)
    
    
    if show_legend:
        if s_var is not None:
            leg1 = plt.legend(*scatter.legend_elements(
                "sizes", num=n_dots_legend, func=lambda x: x*smax/max_size),
                title=s_var, title_fontsize=leg_tit_fontsize,
                fontsize=leg_fontsize, loc=loc_legend_size)
        # optional: loc=4, bbox_to_anchor=(1.2, 0))
        if c_var is not None:
            if pd.api.types.is_string_dtype(df[c_var]):
                h = [plt.plot([],[], color=cu[i], marker="o", 
                              linestyle="None")[0] for i in range(k)]
                plt.legend(handles=h, loc=loc_legend_color, labels=list(vals),
                           title=c_var, fontsize=leg_fontsize,
                           prop={"size": leg_size})
            # optional: loc=4, bbox_to_anchor=(1.2, 1))
        if s_var is not None:
            ax.add_artist(leg1)
    
    if remove_axis:
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    plt.tight_layout()
    
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    
    if keep_just_frame:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    if axis_off:
        plt.axis("off")
    
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
        
    if save_results:
        im = miscbib.BibImage(f_name, name="scatter plot", core=core)
        return im
    return None
        
        
        
def box_plot_from_d(d, boxtype="box", min_len=1, min_sum=0, top=5, 
                    rotate=90, xlabel="", ylabel="",
                    label_dict={}, f_name=None, dpi=1200, fontsize=10, 
                    save_results=True, file_name=None, core=False, **kwds):
    d = sorted(d.items(), key=lambda item: -len(item[1]))
    d = dict([(k,v) for (k,v) in d 
         if sum(v) > min_sum
         if len(v) > min_len][:top])
    if len(label_dict):
        labels = [label_dict[l] for l in d.keys()]
    else:
        labels = d.keys()
    #fig = plt.figure()
    fig, ax = plt.subplots()
    if boxtype == "box":
        plt.boxplot(d.values(), labels=labels, **kwds)
    elif boxtype == "violin":
        plt.violinplot(d.values())
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)
    plt.xticks(fontsize=fontsize)
    plt.xticks(rotation=rotate)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
    if save_results:
        im = miscbib.BibImage(file_name, name=f"box plot sources {ylabel}", core=core)
        return im
    return None
        
# specific

def plot_production(prod_df, years=None, first_axis="Number of documents", 
                    second_axis="Cumulative number of citations", 
                    f_name=None, save_results=True, core=True, dpi=1200, title=None, 
                    comma=False, **kwds):
    has_before=False
    
    if type(prod_df.at[0, "Year"]) not in [np.int64, np.int32, int]: # before
        has_before = True
        before = prod_df.iloc[0]
        before["Year"] = int(before["Year"].split()[1])
        prod_df = prod_df.drop(index=0)
        min_y = prod_df["Year"].min()
        prod_df.iloc[0] = before
    
    if years is None:
        years = prod_df["Year"].min(), prod_df["Year"].max()+1
    else:
        has_before=False
        years[1] += 1
        prod_df = prod_df[prod_df["Year"].between(years[0], years[1], inclusive="left")]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Year")
    plt.xticks(rotation="vertical")
    ax1.set_ylabel(first_axis)
        
    ax1.bar(range(*years), prod_df[first_axis], **kwds)      
    ax1.set_xticks(range(*years))
    if second_axis is not None:
        ax2 = ax1.twinx() 
    if second_axis is not None:
        ax2.set_ylabel(second_axis)
        ax2.plot(range(*years), prod_df[second_axis], color="black")
    if title is not None:
        ax1.set_title(title)  
    if comma:
        import matplotlib as mpl
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
        ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    if has_before:
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        labels[0] = "before " + str(years[0])
        labels[1:] = [str(y) for y in range(*years)[1:]]
        ax1.set_xticklabels(labels)
    
    fig.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
        
    if save_results:
        im = miscbib.BibImage(f_name, name="scientific production", core=core, fig=fig)
        return im
    return fig


def plot_dynamics(df_d, items=[], xlabel="Year", ylabel="Number of documents", 
                  title=None, cum=False, years=None, l_title=None, 
                  f_name=None, save_results=True, core=False, dpi=1200, **kwds):
    if years is not None:
        df_d = df_d[df_d["Year"].between(years[0], years[1])]
    df_d.index = df_d["Year"]
    if len(items) == 0:
        items = [c for c in df_d.columns if c != "Year"]
    if cum:
        items = [it for it in items if "cum" in it]
    else:
        items = [it for it in items if "cum" not in it]
    fig, ax = plt.subplots()
    for it in items:
        label = it if it[:3] != "cum" else it[3:]
        plt.plot(df_d[it], label=label, **kwds)
    ax.set_xlabel(xlabel)
    if cum:
        ylabel = "Cumulative " + ylabel.lower()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.legend(title=l_title)
    
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
        
    if save_results:
        im = miscbib.BibImage(f_name, name="scientific production", core=core)
        return im
    return None

def plot_subset_production(df_inds, df, items=[], exclude=[], y="Cited by", 
                           top=10, names=None, xlabel="", ylabel="", 
                           cmap="plasma", f_name=None,
                           size_one=20,
                           label="Number of citations", dpi=1200,
                           save_results=True, core=False):
    
    if len(items):
        items = [it for it in items if it not in exclude]
        top = len(items)
    else:
        items = df_inds.columns[:top] # tukaj je predpostavka, da so stolpci urejeni glede na 
    
    df_inds = df_inds[items]
    df_inds_y = df_inds.multiply(df[y], axis=0)
    df_vals = pd.concat([df_inds_y, df["Year"]], axis=1).groupby("Year").agg(sum)
    
    cits, years = [], []
    
    # determine smallest number of citations and first year (largest, last)
    for c in df_inds.columns[::-1]:
        cits += [ll for ll in df_vals[c].tolist() if ll!=0] 
        years += df_vals[c][df_vals[c]>0].index.tolist()
    mc, Mc = min(cits), max(cits)
    my, My = min(years), max(years)
    
    fig, ax = plt.subplots()
    plt.tight_layout()
    
    sm = cm.ScalarMappable(norm=Normalize(vmin=mc, vmax=Mc), cmap=cmap)  
    
    for i, c in enumerate(df_inds.columns[::-1]):
        y = df_vals[c][df_vals[c] > 0]
        plt.plot([y.index.min(), y.index.max()], [i, i], 
                 color="k", linewidth=0.5)
        plt.scatter(y.index, [i]*len(y), 
                    sizes=size_one*df["Year"][df_inds[c]==1].value_counts(),
                    color=sm.to_rgba(y))
        
    if names is None:
        names = [c for c in df_inds.columns[::-1]] 
    
    plt.colorbar(sm, label=label, ticks=(np.ceil(mc), np.floor(Mc)))
    ax.set_xticks(range(my, My+1)) 
    ax.set_yticks(range(top))

    ax.set_xticklabels(range(my, My+1), rotation=90) 
    ax.set_yticklabels(names)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
        
    if save_results:
        im = miscbib.BibImage(f_name, name="scientific production", core=core)
        return im
    return None


def plot_distr_citations(df, v="Cited by", min_cist=1, bins=None, nd=None,
                         xlabel="Number of citations", ylabel="Number of documents",
                         title=None, dpi=1200, core=False,
                         f_name=None, save_results=True,
                         **kwds):
    df = df[df[v] >= min_cist]
    fig, ax = plt.subplots()
    if bins is not None:
        plt.hist(df[v], bins=bins, **kwds)
    else:
        if nd is not None:
            bins = int(len(df)/nd)
            plt.hist(df[v], bins=bins, **kwds)
        else:
            plt.hist(df[v], **kwds)
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    fig.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
        
    if save_results:
        im = miscbib.BibImage(f_name, name="distribution of number of citations", core=core)
        return im
    return None


def barh_plot_top(df, by="Number of documents", top=10, name_var=None, core=False, 
                  c_var="Total number of citations", dpi=600, sig_digits=2, file_name=None, save_results=True, color_plot="k",
                  **kwds):   
    
        df_plot = df.sort_values(by, ascending=False)[:top]        
        dig = miscbib.round_properly(by, sig_digits)
        if dig == 0:
            df_plot[by] = df_plot[by].astype(int)
        else:
            df_plot[by] = df_plot[by].round(dig)

        if c_var is not None:
            barh_plot_df(df_plot, name_var, by, c_var=c_var, dpi=dpi, n_digit=dig, 
                                     f_name=file_name, **kwds) # v končni verziji boš dal ldf čez, torej self.ldf(name_var), self.ldf(by)
        else:
            barh_plot_df(df_plot, name_var, by, dpi=dpi, n_digit=dig, 
                                     f_name=file_name, color = color_plot, **kwds)            

        if save_results:
            im = miscbib.BibImage(file_name, name=f"top sources {by}", core=core)
            return im
        return None

def plot_venn(df_ind, subset=None, include_totals=True, title=None, f_name=None,  save_results=True, dpi=600, core=False, **kwds):
    if subset is not None:
        df_ind = df_ind[subset]
    d = {c: set(df_ind.index[df_ind[c]==1].tolist()) for c in df_ind.columns}
    if include_totals:
        d = dict([(str(k) + " (%d)" % len(d[k]), v) for k, v in d.items()])

    venn.venn(d, **kwds)

    if title is not None:
        plt.title(title)
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
        if save_results:
            im = miscbib.BibImage(f_name, name="venn", core=core)
            return im
    

def spectroscopy_plot(years, f_name=None, dpi=600, save_results=True, core=False, **kwds):
    yy = utilsbib.unnest(years)
    ff = sorted(utilsbib.freqs_from_list(yy, order=2))
    x, y = [[f[i] for f in ff] for i in [0,1]]
    fig, ax = plt.subplots()
    plt.plot(x,y, **kwds)
    plt.xlabel("year")
    plt.ylabel("number of cited documents")
    plt.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
    if save_results:
        im = miscbib.BibImage(f_name, name="spectroscopy", core=core)
        return im
    return None




def trend_plot(df_prod, left_var="Q1 year", right_var="Q3 year", mid_var="median year",
               s_var="Number of documents", c_var="Total number of citations",
               main_color="blue", x_label="Year (Q1-median-Q3)",
               l_var=None, max_size=100, cmap="plasma",
               f_name=None, dpi=1200, save_results=True, core=False):
    fig, ax = plt.subplots()

    for index, row in df_prod.iterrows():
        plt.plot([row[left_var], row[right_var]], [index, index], 
                 color="k", linewidth=0.5)
    
    if s_var is not None:
        sizes = df_prod[s_var] * max_size / df_prod[s_var].max()
    else:
        sizes = [max_size] * len(df_prod)
    if c_var is not None:
        mc, Mc = df_prod[c_var].min(), df_prod[c_var].max()
        sm = cm.ScalarMappable(norm=Normalize(vmin=mc, vmax=Mc), cmap=cmap) 
        colors = [sm.to_rgba(y) for y in df_prod[c_var]]
    else:
        colors = main_color

    plt.scatter(df_prod[mid_var], df_prod.index, sizes=sizes, color=colors)

    if c_var is not None:
        plt.colorbar(sm, label=c_var, ticks=(np.ceil(mc), np.floor(Mc)))
    
    if x_label is not None or x_label != "":
        ax.set_xlabel(x_label)

    if l_var is not None:
        ax.set_yticks(range(len(df_prod)))
        ax.set_yticklabels(df_prod[l_var], fontsize=int(300/len(df_prod)))
        ax.set_ylabel(l_var)
    plt.tight_layout()
    
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
    
    if save_results:
        im = miscbib.BibImage(f_name, name=f"trend plot {f_name}", core=core)
        return im
    return None


def plot_country_collab_production(df, s_var="SCP", m_var="MCP",
                            l_var="Country of corresponding author", top=10,
                            f_name=None, dpi=1200,
                            save_results=True, core=False, **kwds):
    fig = plt.figure()
    
    nn = df["MCP ratio"].head(top).iloc[::-1].tolist()
    nn = [np.round(nnn, 3) for nnn in nn]


    df = df[[l_var, s_var, m_var]].head(top).iloc[::-1]
    df = df.set_index(l_var)
    ax = df.plot(kind="barh", stacked=True, **kwds)
    
    #df = df.reset_index()
    #for index, row in df.iterrows():
    #    print((nn[index], (row[s_var]+row[m_var], index/top)))
    #    ax.annotate(nn[index], (row[s_var]+row[m_var], index/top))
    
    i = 0
    for p in ax.patches:
        if p.get_x() != 0:
            ax.text(p.get_width() + p.get_x() +0.1, p.get_y()+0.1, nn[i])
            i += 1 # tole je čisto narobe (kar se tiče številk)

    
    plt.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")
        plt.savefig(f_name + ".pdf")
    if save_results:
        im = miscbib.BibImage(f_name, name="country collaboration production", core=core)
        return im
    return None

# specific plot - shows differences between two series in percentages

def plot_props_df(props_df, v="PP difference", pos_color="lightblue", neg_color="lightcoral",
                  name_var="name", x_label=None, 
                  show_0_line=True, p_lines=[], color_dict=None, color_var=None, 
                  padding = (20,20), show_col_leg=True, leg_loc="upper center", bbox_to_anchor=(0.3, -0.2),
                  leg_n_col=3, font_size_data=8, font_size_labels=8, y_label=None,
                  f_name=None, dpi=1200, **kwds):
    # p_lines daš recimo [5,-5]
    # za color_var boš dal "perspective"
    
    props_df = props_df.sort_values(by=v)
    colors = [pos_color if val >= 0 else neg_color for val in props_df[v]]
    
    if name_var is not None:
        props_df = props_df.set_index(name_var)
    
    bars = plt.barh(props_df.index, props_df[v], color=colors) 

    if show_0_line:
        plt.axvline(0, color="black", linestyle='-')
    
    if y_label is not None:
        plt.ylabel(y_label)
        
    plt.yticks(fontsize=font_size_labels)
    
    for p in p_lines:
        plt.axvline(p, color="gray", linestyle='--')  
    
    if x_label is None:
        d_label = {"PP difference": "Difference (percentage points)", "prop %": "Percentage", "P difference": "Difference (percentage)"}
        if v in d_label:
            x_label = d_label[v]
    if x_label is not None:
        plt.xlabel(x_label)

    for bar, value in zip(bars, props_df[v]):
        if value >= 0:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", ha="left", va="center", color="black", fontsize=font_size_data)
        else:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", ha="right", va="center", color="black", fontsize=font_size_data)

    plt.xlim(props_df[v].min()-padding[0], props_df[v].max()+padding[1])

    if (color_dict is not None) and (color_var is not None):
        for i, (tick_label, color) in enumerate(zip(plt.yticks()[1], props_df[color_var])):
            tick_label.set_color(color_dict[color])

        handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[color], label=color) for i, color in enumerate(props_df[color_var].unique())]
        
        if show_col_leg:
            plt.legend(handles=handles, title=color_var.capitalize(), loc=leg_loc, bbox_to_anchor=bbox_to_anchor, ncol=int(len(props_df[color_var].unique())/leg_n_col))

    plt.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".svg")  
        plt.savefig(f_name + ".pdf")

# network plots

nd = {"max_size": 100, "max_label_font": 20, "min_label_font": 5, "hide_small_font": False,
              "labels_font": 6, "rad": 0.3, "max_edge_width": 10,
              "layout": nx.spring_layout, "cmap": "viridis",
              "d_cmap": "Set1", "c_lab": None, "digits": 3,
              "color_edges": 1, "adjust_labels": False,
              "arrowprops": dict(arrowstyle="-", color="k", alpha=.5)}

    
def plot_network(G, part=None, vec_color=None, vec_size=None, nd=nd, 
                 edge_weight="weight", f_name=None, dpi=600, **kwds):

    for k, v in kwds.items():
        nd.update({k: v})
    fig, ax = plt.subplots()
    pos = nd["layout"](G)
    
    #node_colors = [(0,0,0,1)] * len(G)
    node_colors = "b"
    
    if part is not None:
        if vec_color is not None:
            print("Two possibilities for colors. Network will be plotted\
                  by partition. To color nodes using vec_color, \
                      please set part=None.")
        scalarmappaple = cm.ScalarMappable(cmap=nd["d_cmap"])
        c = pd.Series(part)
        c = c.astype(str)
        pv = list(part.values())
        scalarmappaple.set_array(pv)   
    elif vec_color is not None:
        scalarmappaple = cm.ScalarMappable(cmap=nd["cmap"])
        c = pd.Series(vec_color)
        scalarmappaple.set_array(vec_color)
        
    if (part is not None) or (vec_color is not None):
       
        if pd.api.types.is_numeric_dtype(c):
            if c.max() - c.min() >= 1:
                plt.colorbar(scalarmappaple, label=nd["c_lab"], ax=ax,
                             ticks=(np.ceil(c.min()), np.floor(c.max())))           
            else:
                plt.colorbar(scalarmappaple, label=nd["c_lab"],
                             ticks=(np.round(c.min(), nd["digits"]),
                                    np.round(c.max(), nd["digits"])))
            node_colors = scalarmappaple.to_rgba(c)
        elif pd.api.types.is_string_dtype(c):
            scalarmappaple = cm.ScalarMappable(cmap=nd["d_cmap"])
            cmap = nd["d_cmap"]
            vals = c.unique()
            k = len(vals)
            cd = dict(zip(*(vals, scalarmappaple.cmap(np.linspace(0,1,k)))))
            c = [cd[v] for v in list(c)]
            node_colors = c
    else:
        c = None      
    
    if len(node_colors) == 1:
        node_colors = [node_colors]*len(G)
    node_colors_d = dict(zip(*(list(G.nodes), node_colors)))
    
    if vec_size is not None:
        M = max(vec_size)
        vec_size = [nd["max_size"]/M*s for s in vec_size]
    else:
        vec_size = [nd["max_size"]] * len(G)
    if nd["max_label_font"] is not None:
        M = max(vec_size)
        labels_font = [int(max(nd["max_label_font"]/M*s, nd["min_label_font"])) for s in vec_size]
        if nd["hide_small_font"]:
            def rep(x): 
                if x == nd["min_label_font"]:
                    return 0
                return x
            labels_font = list(map(rep, labels_font))
        d = dict([(list(G.nodes)[i], labels_font[i]) for i in range(len(G))])
        
    nx.draw_networkx_nodes(G, pos, node_size=vec_size, node_color=c)
    
    if nd["adjust_labels"]:
        texts = [plt.text(pos[key][0], pos[key][1], key, fontsize=d[key]) for key in pos.keys()]
        adjust_text(texts, arrowprops=nd["arrowprops"])
    else:
        if nd["max_label_font"] is None:
            nx.draw_networkx_labels(G, pos, font_size=labels_font)
        else:
            for node, (x, y) in pos.items():
                plt.text(x, y, node, fontsize=d[node], ha="center", va="center")
            
    edge_weights = [d[edge_weight] for _, _, d in G.edges(data=True)]
    Mw = max(edge_weights)
    
    for edge in G.edges(data=True):
        source, target, w = edge
        rad = nd["rad"]
        if nd["color_edges"] is not None:
            try:
                color = 0.5 * (node_colors_d[source] + node_colors_d[target])
            except:
                color = "black"
        else:
            color = "black"

        arrowprops=dict(arrowstyle="-", color=color,  connectionstyle=f"arc3,rad={rad}",
                               alpha=0.4, linewidth=nd["max_edge_width"]/Mw*w[edge_weight])
        ax.annotate("", xy=pos[source], xytext=pos[target], arrowprops=arrowprops)           
    
    #plt.legend(part) # legenda za barve, če imaš particijo - to še ne dela
    
    if f_name is not None:
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f_name+".png", dpi=dpi)
        plt.savefig(f_name+".pdf")
        plt.savefig(f_name+".svg")
        
# wordcloud

from wordcloud import WordCloud, get_single_color_func

class GroupedColorFunc(object): # from https://stackoverflow.com/questions/70883110/python-wordcloud-how-to-make-the-word-colour-based-on-a-data-column
    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
        
def plot_word_cloud_from_freqs(df_perf, name_var="word", freq_var="f", color_var="Average year of publication",
                          background_color="white", f_name=None, dpi=600, round_scale=True, 
                          label_color_var=None, **kwds):
    
    d = dict(zip(df_perf[name_var], df_perf[freq_var]))

    wc = WordCloud(background_color=background_color, **kwds)
    wc.generate_from_frequencies(frequencies=d)
    
    if color_var is not None:  
        c, w = list(df_perf[color_var]), list(df_perf[name_var])
    
        scalarmappaple = cm.ScalarMappable(cmap="viridis")
        scalarmappaple.set_array(c)
        colors = [matplotlib.colors.rgb2hex(cc) for cc in scalarmappaple.to_rgba(c)]
    
        s = [(colors[i],w[i]) for i in range(len(df_perf))]
        d = defaultdict(list)
        for k, v in s:
            d[k].append(v)  
    
        grouped_color_func = GroupedColorFunc(d, "grey")
        wc.recolor(color_func=grouped_color_func)

    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    if label_color_var is None:
        label_color_var = color_var

    if color_var is not None:
        if round_scale:
            plt.colorbar(scalarmappaple, label=label_color_var, ticks=(np.ceil(min(c)), np.floor(max(c))))
        else:
            plt.colorbar(scalarmappaple, label=label_color_var)
    
    
    if f_name is not None:
        plt.savefig(f_name+".png", dpi=dpi)
        plt.savefig(f_name+".svg")
        plt.savefig(f_name+".pdf")
        
# plots for cca, pls, etc.

def loadings_plot(df_l1, df_l2, c1=0, c2=1, min_norm=0.1, col1="r", col2="b", l1="X", l2="Y", show_grid=True, f_name=None, dpi=600):
    def norm(x,y):
        return x**2+y**2

    X, Y = df_l1.to_numpy(), df_l2.to_numpy()
    
    plt.figure(figsize=(10, 10))
    for i in range(X.shape[0]):
        x, y = X[i, c1], X[i, c2]
        if norm(x,y) < min_norm:
            continue
        plt.arrow(0, 0, x, y, color=col1, alpha=0.5)
        plt.text(x,y, df_l1.index[i], color=col1)

    for i in range(Y.shape[0]):
        x, y = Y[i, c1], Y[i, c2]
        if norm(x,y) < min_norm:
            continue
        plt.arrow(0, 0, x, y, color=col2, alpha=0.5)
        plt.text(x,y, df_l2.index[i], color=col2)
        
    if show_grid:
        plt.grid(True, linestyle="--")
    
    patch1 = plt.Line2D([0], [0], color=col1, linewidth=2, label=l1)
    patch2 = plt.Line2D([0], [0], color=col2, linewidth=2, label=l2)
    
    plt.xlabel(f"Component {c1+1}")
    plt.ylabel(f"Component {c2+1}")
    
    plt.legend(handles=[patch1, patch2])
    
    if f_name is not None:
        plt.savefig(f_name+".png", dpi=dpi)
        plt.savefig(f_name+".svg")
        plt.savefig(f_name+".pdf")
        
# sankey

import plotly.graph_objects as go

def plot_sankey(P, labels, color_vals, ks, font_size=10, f_name=None, save_html=False, c_names=None, cmap="plasma", single_color="black"):
    k = len(ks)
    colors = cvals_to_rgb(color_vals, cmap=cmap, single_color=single_color)
    x = []
    for i in range(k):
        xx = [i/(k-1)]*ks[i]
        x += xx
    fig = go.Figure(data=[
        go.Sankey(arrangement="snap",
                  node=dict(pad=10, thickness=10, label=labels, 
                            color=colors, x=x, y=[0.1]*len(x)),
                  link = dict(source=P["source"], target=P["target"],
                              value=P["value"]))])
    fig.update_layout(title_text="%d - fields plot" % k, 
                  font=dict(color="black"), font_size=10)
    #Tu se bo poslovenilo
    #fig.update_layout(title_text="Sankeyjev prikaz", font=dict(color="black"), font_size=10)
    fig.update_traces(textfont_size=font_size)
    if c_names is None:
        c_names =["Text %d" % (i+1) for i in range(k)]
    # Tukaj se bo delalo poslovenjanje
    #c_names = ["ključne besede", "opazovani koncepti", "viri (revije)"]
    for i, l in enumerate(c_names):
        fig.add_annotation(x=i/(k-1), y=1.1,
                text=c_names[i], showarrow=False)

    if f_name is not None:
        fig.write_image(f_name+".png", scale=5)
        fig.write_image(f_name+".svg")
    if save_html:
        fig.write_html(f_name+".html")
        
def cvals_to_rgb(color_vals, 
                 cmap="plasma", single_color="black", cbar_label="", dpi=600):
    
    if len(color_vals):
        scalarmappaple = cm.ScalarMappable(cmap=cmap)
        scalarmappaple.set_array(color_vals)
        color_array = scalarmappaple.to_rgba(color_vals)[:,:-1]
        colors = []
        for c in color_array:
            s = "rgb(" + str(c[0]) + "," + str(c[1]) + "," + str(c[2]) + ")"
            colors.append(s)
        
    return colors

# treemap

import squarify

def plot_treemap(perf_df, size_var="Number of documents", color_var="Average year of publication",
                   labels_var="Keyword", show=None, cmap_name="viridis", ndig=2, font_size=10, label_color_var=None, f_name="treemap", dpi=600):
    sizes = perf_df[size_var] / perf_df[size_var].sum()

    value_min = perf_df[color_var].min()
    value_max = perf_df[color_var].max()
    norm = plt.Normalize(vmin=value_min, vmax=value_max)
    colors = cm.get_cmap(cmap_name)(norm(perf_df[color_var]))

    if show is not None:
        perf_df = perf_df.nlargest(show, size_var)
    
    fig, ax = plt.subplots()
    perf_df[labels_var] = perf_df[labels_var].map(lambda x: "\n".join(x.split()))
    squarify.plot(sizes=sizes, label=perf_df[labels_var], color=colors, text_kwargs={"fontsize" :  font_size})

    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    label_color_var = label_color if label_color_var is None else "blue"
    cbar = plt.colorbar(sm, ticks=[value_min, value_max], label=label_color_var)
    
    cbar.ax.set_yticklabels([np.round(value_min,ndig), np.round(value_max,ndig)])

    plt.axis("off")
    if f_name is not None:
        plt.savefig(f_name+".png", dpi=dpi)
        plt.savefig(f_name+".svg")
        plt.savefig(f_name+".pdf")
        
        
# plot group production

def plot_production_groups(prod_df, vrs1, vrs2=None, kind="bar", labels=None, colors=None, years=None,
                           f_name=None, dpi=1200, title=None, x_label=None, y_label=None, y_label_2=None,
                           before_str="before "):
    
    
    has_before=False
    if type(prod_df.at[0, "Year"]) not in [np.int64, np.int32, np.float64, int]: # before
        has_before = True
        before = prod_df.iloc[0]
        before["Year"] = int(before["Year"].split()[1])
        prod_df = prod_df.drop(index=0)
        min_y = prod_df["Year"].min()
        prod_df.iloc[0] = before

    
    if years is None:
        years = prod_df["Year"].min(), prod_df["Year"].max()+1
    else:
        has_before=False
        years[1] += 1
        prod_df = prod_df[prod_df["Year"].between(years[0], years[1], inclusive="left")]
    
    fig, ax = plt.subplots()
    bottom, y_var = 0, 0
    
    n = len(vrs1)
    if labels is None:
        labels = [""] * n
        
    if kind == "bar":
    
        for i in range(n):
            if colors is not None:
                ax.bar(prod_df["Year"], prod_df[vrs1[i]], bottom=bottom, label=labels[i], color=colors[i])
            else:
                ax.bar(prod_df["Year"], prod_df[vrs1[i]], bottom=bottom, label=labels[i])
            bottom += prod_df[vrs1[i]]
            if vrs2 is not None:
                if "Proportion of documents in group" not in vrs2:
                    y_var += prod_df[vrs2[i]]
    
    elif kind == "line":
            
        for i in range(n):
            if colors is not None:
                ax.plot(prod_df["Year"], prod_df[vrs1[i]], label=labels[i], color=colors[i])
            else:
                ax.plot(prod_df["Year"], prod_df[vrs1[i]], label=labels[i])
    
    
    
    if vrs2 is not None:
        ax2 = ax.twinx()
        if "Proportion of documents in group" not in vrs2:
            if y_label_2 is None:
                ax2.set_ylabel("Cumulative number of citations")
            else:
                ax2.set_ylabel(y_label_2)
            ax2.plot(prod_df["Year"], y_var, color="black")
        else:
            ax2.set_ylabel(vrs2)
            ax2.plot(prod_df["Year"], prod_df[vrs2], color="black")

    if x_label is None:
        ax.set_xlabel("Year")
    else:
        ax.set_xlabel(x_label)
    if y_label is None:
        ax.set_ylabel("Number of documents")
    else:
        ax.set_ylabel(y_label)
    
    if title is not None:
        ax.set_title(title)
    
    if has_before:
        ax.set_xticks(range(*years))
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = before_str + str(years[0])
        labels[1:] = [str(y) for y in range(*years)[1:]]
        ax.set_xticklabels(labels, rotation=90)
    
    legend = ax.legend()
    
    # Če bi želel spreminjati velikost pisave na legendi
    #for text in legend.get_texts():
    #    text.set_fontsize(8)
    
    
    fig.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".pdf")
        plt.savefig(f_name + ".svg")



def plot_stacked_barh(df, top=5, x="Number of documents", label="Label", group="group",
                     show_percentages=True, perc_color="black", group_colors=None, title=None,
                     f_name="top_bars_stacked", dpi=600):
    

    pivot_df = df.pivot_table(index=label, columns=group, values=x, aggfunc="sum", fill_value=0)

    if top is not None:
        top_labels = pivot_df.sum(axis=1).sort_values(ascending=False).head(top).index
    else:
        top_labels = pivot_df.sum(axis=1).sort_values(ascending=False).index
    sorted_df = pivot_df.loc[top_labels]

    percentage_df = (sorted_df / sorted_df.sum(axis=1).values.reshape(-1, 1)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    default_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if group_colors is None:
        colors = default_colors
    else:
        colors = [group_colors.get(group_name, None) for group_name in sorted_df.columns]

    sorted_df.plot(kind="barh", stacked=True, ax=ax, color=colors)

    if title is not None:
        plt.title(title)
    plt.xlabel(x)
    plt.ylabel(label)
    plt.legend(title=group, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.grid(False)

    ax.invert_yaxis()

    if show_percentages:
        for i, (index, row) in enumerate(sorted_df.iterrows()):
            x_offset = 0
            for col in sorted_df.columns:
                value = row[col]
                if value != 0:
                    percentage = percentage_df.loc[index, col]
                    ax.text(x_offset + value / 2, i, f"{percentage:.1f}%", ha="center", va="center", color=perc_color, fontweight="bold")
                x_offset += value

    fig.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".pdf")
        plt.savefig(f_name + ".svg")
        
    return percentage_df

  
def create_grouped_top_k_barh_chart(df, variable_column="selected_variable", group_column="group", label_var="name",
                                    k=5, group_colors=None, x_label=None, y_label=None, chart_title=None, ncar=50,
                                    f_name="top_bars", dpi=600, **kwds):

    df = df.sort_values(by=variable_column, ascending=False)
    if group_colors is not None:
        group_colors = [group_colors[c] for c in df[group_column].unique()]

    top_k_df = df.groupby("group", group_keys=False).apply(lambda x: x.nlargest(k, variable_column))
    top_k_df[label_var] = top_k_df[label_var].apply(lambda x: "\n".join(wrap(x, width=ncar)))

    custom_order = df[group_column].value_counts().index.tolist()[::-1]
    top_k_df["tmp"] = pd.Categorical(top_k_df[group_column], categories=custom_order, ordered=True)
    top_k_df = top_k_df.sort_values(["tmp", variable_column], ascending=False)

    fig, ax = plt.subplots()

    top_k_df = utilsbib.handle_duplicates(top_k_df, name_var=label_var)

    sns.set(style="whitegrid")
    
    sns.barplot(x=variable_column, y=label_var, hue=group_column, data=top_k_df, dodge=False, palette=group_colors, **kwds) 

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if chart_title is not None:
        plt.title(chart_title)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right")

    fig.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".pdf")
        plt.savefig(f_name + ".svg")
  

# plot dataframes with associations

def split_name_(x, s):
    s += " "
    if s in x:
        return x.split(s)[1]
    return x


def plot_assocations_df(df, measure="yule_Q", name="x", order=None, cmap="viridis", annot=True, fmt=".2g", label_cbar=None, 
               fmt_cbar="%.2g", xticklabels=True, yticklabels=True,
               xlabel=None, ylabel=None, f_xlabel=10, f_ylabel=10,
               f_colorbar=10, f_lab_colorbar=10,
               annot_size=10, ylabel_size=10, xlabel_size=10,
               title=None, f_name=None, dpi=1200, **kwds):
    
    
    
    df = df[[name] + [c for c in df.columns if measure in c]]
    df = df.set_index(name)
    df = df.rename(columns=lambda x: split_name_(x, measure))
    
    if order is not None:
        df = df[order]
    
    heatmap_df(df, cmap=cmap, annot=annot, fmt=fmt, label_cbar=label_cbar, 
                   fmt_cbar=fmt_cbar, xticklabels=xticklabels, yticklabels=yticklabels,
                   xlabel=xlabel, ylabel=ylabel, f_xlabel=f_xlabel, f_ylabel=f_ylabel,
                   f_colorbar=f_colorbar, f_lab_colorbar=f_lab_colorbar,
                   annot_size=annot_size, ylabel_size=ylabel_size, xlabel_size=xlabel_size,
                   prepocess_columns=False, title=title, f_name=f_name, dpi=dpi, **kwds)
    


def plot_lotka_law(n_values, f_values, results, ylabel="Number of authors", f_name=None, dpi=600, **kwds):
    C = results['C']
    n_param = results['n_param']
    predicted_values = results['predicted_values']
    residuals = results['residuals']

    # Plot the data and the fitted curve
    plt.figure()
    plt.scatter(n_values, f_values, label='Data')
    plt.plot(n_values, predicted_values, label='Fitted Lotka\'s Law',  **kwds)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of ddocuments')
    plt.ylabel(ylabel)
    plt.legend()
    if f_name:        
        plt.savefig(f"{f_name}_fit.png", dpi=dpi)
        plt.savefig(f"{f_name}_fit.pdf")
        plt.savefig(f"{f_name}_fit.svg")


    # Plot residuals
    plt.figure()
    plt.scatter(n_values, residuals)
    plt.hlines(0, xmin=min(n_values), xmax=max(n_values), **kwds)
    plt.xscale('log')
    plt.xlabel('Number of documents')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    if f_name:
        plt.savefig(f"{f_name}_residuals.png", dpi=dpi)
        plt.savefig(f"{f_name}_residuals.pdf")
        plt.savefig(f"{f_name}_residuals.svg")


    # Calculate the empirical cumulative distribution function (ECDF)
    def ecdf(data):
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n+1) / n
        return x, y

    # Empirical CDF of observed data
    x_empirical, y_empirical = ecdf(f_values)

    # Theoretical CDF based on fitted Lotka's Law
    x_theoretical = np.sort(f_values)
    y_theoretical = predicted_values.cumsum()
    y_theoretical /= list(y_theoretical)[-1]  # Normalize to make it a CDF

    # Plot the ECDF and theoretical CDF
    plt.figure()
    plt.step(x_empirical, y_empirical, label='Empirical CDF', where='post')
    plt.step(x_theoretical, y_theoretical, label='Theoretical CDF', where='post', linestyle='--', **kwds)
    plt.xlabel('Number of documents')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    if f_name:
        plt.savefig(f"{f_name}_cdf.png", dpi=dpi)
        plt.savefig(f"{f_name}_cdf.pdf")
        plt.savefig(f"{f_name}_cdf.svg")


from scipy.cluster.hierarchy import dendrogram, fcluster

# clustering

def plot_dendrogram(linkage_matrix, df, num_clusters=None, f_name=None, dpi=600):
    """
    Plot dendrogram and optionally determine the optimal number of clusters.
    
    Parameters:
    linkage_matrix (np.ndarray): Linkage matrix.
    df (pd.DataFrame): Original dataframe with names.
    num_clusters (int): Number of clusters to determine and label instances.
    
    Returns:
    pd.DataFrame: DataFrame with labels if num_clusters is specified.
    """
    plt.figure(figsize=(10, 7))
    dendro = dendrogram(linkage_matrix, labels=df.iloc[:, 0].values)
    plt.title("Dendrogram")
    plt.xlabel("Index or (Cluster Size)")
    plt.ylabel("Distance")
    if f_name:
        plt.savefig(f"{f_name}_cdf.png", dpi=dpi)
        plt.savefig(f"{f_name}_cdf.pdf")
        plt.savefig(f"{f_name}_cdf.svg")
    
    if num_clusters:
        labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        df['Cluster'] = labels
        return df
    return None

def hierarchical_clustering_pipeline(df, dissimilarity_metric='jaccard', linkage_method='ward', num_clusters=None, f_name=None, dpi=600):
    """
    Complete pipeline to compute dissimilarity matrix, perform hierarchical clustering,
    plot dendrogram and optionally determine the optimal number of clusters.
    
    Parameters:
    df (pd.DataFrame): Binary dataframe with first column as names.
    dissimilarity_metric (str): Dissimilarity metric. Options: 'jaccard', 'hamming', 'euclidean'.
    linkage_method (str): Linkage method. Options: 'single', 'complete', 'average', 'ward'.
    num_clusters (int): Number of clusters to determine and label instances.
    
    Returns:
    pd.DataFrame: DataFrame with labels if num_clusters is specified.
    """
    dist_matrix = utilsbib.compute_dissimilarity_matrix(df, metric=dissimilarity_metric)
    linkage_matrix = utilsbib.perform_hierarchical_clustering(dist_matrix, method=linkage_method)
    result_df = plot_dendrogram(linkage_matrix, df, num_clusters=num_clusters, f_name=f_name, dpi=dpi)
    return result_df

    
    """
    df = df[[name] + [c for c in df.columns if measure in c]]
    df = df.rename(columns=lambda x: split_name_(x, measure))

    df = df.set_index(name)
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, cmap=cmap, cbar_kws={'label': c_label})

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    fig.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".pdf")
        plt.savefig(f_name + ".svg")
    """
    
"""
def create_grouped_top_k_barh_chart(df, variable_column="selected_variable", group_column="group", label_var="name",
                                    k=5, group_colors=None, x_label=None, y_label=None, chart_title=None, ncar=50,
                                    f_name="top bars", dpi=600):


    # Sort the DataFrame by the selected variable in descending order
    df = df.sort_values(by=variable_column, ascending=False)

    # Initialize a figure and axis for the bar chart
    fig, ax = plt.subplots()

    # Group the data by the 'group' column and reverse the order
    grouped_data = df.groupby(group_column)
    group_names = list(reversed(df[group_column].unique()))

    # Initialize the color cycle based on the provided group_colors or default colors
    if group_colors is None:
        group_colors = {group: plt.get_cmap("tab10")(i) for i, group in enumerate(group_names)}


    for group in group_names:
        group_data = grouped_data.get_group(group)
        # Get the top k values for the current group and reverse their order
        top_k_values = group_data.nlargest(k, variable_column)[::-1]
        
        # Wrap the 'name' column text
        top_k_values[label_var] = top_k_values[label_var].apply(lambda x: "\n".join(wrap(x, width=ncar)))

        # Plot the horizontal bars for the group
        ax.barh(top_k_values[label_var], top_k_values[variable_column], color=group_colors[group], label=group)


    # Set axis labels and chart title
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if chart_title is not None:
        plt.title(chart_title)

    # Display the legend in the lower-right corner
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc="lower right")

    # Show the chart
    fig.tight_layout()
    if f_name is not None:
        plt.savefig(f_name + ".png", dpi=dpi)
        plt.savefig(f_name + ".pdf")
        plt.savefig(f_name + ".svg")

"""
