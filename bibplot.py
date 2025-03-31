# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:42:20 2023

@author: Lan
"""

from bibanalysis import BiblioStats
import plotbib, miscbib, utilsbib
import numpy as np

class BiblioPlot(BiblioStats):
    
    def plot_top_authors_production(self, items=[], exclude=["nan"],
                                    y="Cited by", top=10, names=None, 
                                    xlabel="Year", ylabel="Author",
                                    size_one=20, label="Number of citations", 
                                    core=False):
        
        im = plotbib.plot_subset_production(self.authors_df_ind, self.df,
                                       items=items, exclude=exclude, y=y, 
                           top=top, names=names, xlabel=xlabel, ylabel=ylabel, 
                           cmap=self.cmap, f_name=self.res_folder+"\\plots\\top authors production",
                           size_one=size_one,
                           label=label, dpi=self.dpi,
                           save_results=self.save_results, core=core)
        if im is not None:
            self.images.append(im)
            def plot_top_authors_production(self, items=[], exclude=["nan"],
                                            y="Cited by", top=10, names=None, 
                                            xlabel="Year", ylabel="Author",
                                            size_one=20, label="Number of citations", 
                                            core=False):                        
                
                im = plotbib.plot_subset_production(self.authors_df_ind, self.df,
                                               items=items, exclude=exclude, y=y, 
                                   top=top, names=names, xlabel=xlabel, ylabel=ylabel, 
                                   cmap=self.cmap, f_name=self.res_folder+"\\plots\\top authors production",
                                   size_one=size_one,
                                   label=label, dpi=self.dpi,
                                   save_results=self.save_results, core=core)
                if im is not None:
                    self.images.append(im)


    def plot_top_ca_countries_production(self, items=[], exclude=["nan"],
                                    y="Cited by", top=10, names=None, 
                                    xlabel="Year", ylabel="Country of corresponding author",
                                    size_one=20, label="Number of citations", 
                                    core=False):
        
        im = plotbib.plot_subset_production(self.ca_countries_df_ind, self.df,
                                       items=items, exclude=exclude, y=y, 
                           top=top, names=names, xlabel=xlabel, ylabel=ylabel, 
                           cmap=self.cmap, f_name=self.res_folder+"\\plots\\top ca countries production",
                           size_one=size_one,
                           label=label, dpi=self.dpi,
                           save_results=self.save_results, core=core)
        if im is not None:
            self.images.append(im)
            def plot_top_authors_production(self, items=[], exclude=["nan"],
                                            y="Cited by", top=10, names=None, 
                                            xlabel="Year", ylabel="Country of corresponding author",
                                            size_one=20, label="Number of citations", 
                                            core=False):
                im = plotbib.plot_subset_production(self.authors_df_ind, self.df,
                                               items=items, exclude=exclude, y=y, 
                                   top=top, names=names, xlabel=xlabel, ylabel=ylabel, 
                                   cmap=self.cmap, f_name=self.res_folder+"\\plots\\top ca countries production",
                                   size_one=size_one,
                                   label=label, dpi=self.dpi,
                                   save_results=self.save_results, core=core)
                if im is not None:
                    self.images.append(im)   
                    
    def plot_distr_citations(self, v="Cited by", min_cist=1, bins=None, nd=None,
                         xlabel="Number of citations", ylabel="Number of documents",
                         title=None, core=False, **kwds):
        im = plotbib.plot_distr_citations(
            self.df, v=v, min_cist=min_cist, bins=bins, nd=nd,
            xlabel=xlabel, ylabel=ylabel, title=title, dpi=self.dpi, core=core, 
            f_name=self.res_folder+"\\plots\\distribution of citations", 
            save_results=self.save_results, **kwds)
        if im is not None:
            self.images.append(im)
    
    def plot_production(self, years=None, first_axis="Number of documents", 
                    second_axis="Cumulative number of citations", title=None,
                    comma=False, **kwds):
        if not hasattr(self, "production_df"):
            self.get_production()
        im = plotbib.plot_production(self.production_df, years=years, first_axis=first_axis, 
                    second_axis=second_axis, f_name=self.res_folder+"\\plots\\scientific production",
                              save_results=self.save_results, dpi=self.dpi,
                              title=title, comma=comma, **kwds)
        if isinstance(im, miscbib.BibImage):
            self.images.append(im)
        return im.fig
    
    # bar-plots
    
    def bar_plot_top_sources(self, by="Number of documents",
                             top=10, name="short", core=False, 
                             c_var="Total number of citations", **kwds):
        # dodaj wrap text in poglej, zakaj cmap ne deluje
        name_var = {"full": "Source title", "short": "Abbreviated Source Title"}[name]
        im = plotbib.barh_plot_top(self.sources_df, by=by, top=top, name_var=name_var, core=core,
                              c_var=c_var, dpi=self.dpi, sig_digits=self.sig_digits, 
                              file_name=self.res_folder+f"\\plots\\top sources by {by}",
                              save_results=self.save_results, color_plot=self.color_plot, **kwds)
        if im is not None:
            self.images.append(im)
            
    def bar_plot_top_ca_countries(self, by="Number of documents",
                             top=10, name_var="CA Country", core=False, 
                             c_var="Total number of citations", **kwds):
        im = plotbib.barh_plot_top(self.ca_countries_df, by=by, top=top, name_var=name_var, core=core,
                              c_var=c_var, dpi=self.dpi, sig_digits=self.sig_digits, 
                              file_name=self.res_folder+f"\\plots\\top CA countries by {by}",
                              save_results=self.save_results, color_plot=self.color_plot, **kwds)
        if im is not None:
            self.images.append(im)
            
    def bar_plot_top_authors(self, by="Number of documents",
                             top=10, name_var="Author", core=False, 
                             c_var="Total number of citations", **kwds):
        im = plotbib.barh_plot_top(self.authors_df, by=by, top=top, name_var=name_var, core=core,
                              c_var=c_var, dpi=self.dpi, sig_digits=self.sig_digits, 
                              file_name=self.res_folder+f"\\plots\\top authors by {by}",
                              save_results=self.save_results, color_plot=self.color_plot, **kwds)
        if im is not None:
            self.images.append(im)
            
    
    # lahko dodaš še keywords, word_and_phrases
    
    # lollipot-plots
    
    def lollipot_plot_top_sources(self, by="Number of documents",
                             top=10, name="short", core=False, 
                             c_var="H-index", **kwds):
        # dodaj wrap text in poglej, zakaj cmap ne deluje
        name_var = {"full": "Source title", "short": "Abbreviated Source Title"}[name]
        im = plotbib.lollipop_plot_df(self.sources_df, by, name_var, c_var=c_var, cmap=self.cmap, 
                                      def_color=self.color_plot, file_name=self.res_folder+f"\\plots\\lolipot top sources by {by}",
                                      dpi=self.dpi, **kwds)
        
        if im is not None:
            self.images.append(im)
    
    # lahko dodaš še ostale
    
    
    
    # scatter-plots
    
    def scatter_plot_top_sources(self, items=[], exclude=["nan"], top=10, min_freq=1, 
                                 by="Number of documents", x="Number of documents", y="Total number of citations",
                                 s="H-index", c="Average year of publication", l="Abbreviated Source Title",
                                 max_size=100, x_scale="log", y_scale="log", max_text_len=30,
                                 show_mean_line=False, show_mu=False, show_reg_line=False,
                                 arrowprops=None, core=True,                                
                                 kwds1={}, kwds2={}):
        items = miscbib.select_items(self.sources_df, "Source title", top=top, min_freq=min_freq, items=items, exclude=exclude)
        df = self.sources_df.loc[self.sources_df["Source title"].isin(items)]

        f_name = self.res_folder +"\\plots\\top sources" if self.save_results else None

        plotbib.scatter_plot_df(df, x, y, s_var=s, c_var=c, l_var=l,
                                 max_size=max_size, cmap=self.cmap, max_text_len=max_text_len,
                                 d_cmap=self.d_cmap, x_scale=x_scale, y_scale=y_scale,
                                 f_name=f_name,
                                 show_mean_line=show_mean_line, show_mu=show_mu, show_reg_line=show_reg_line,
                                 arrowprops=arrowprops, core=core)

        if self.save_results:
            im = miscbib.BibImage(self.res_folder + "\\plots\\top sources", 
                     name="top sources")
            self.images.append(im)

    def scatter_plot_top_ca_countries(self, items=[], exclude=["nan"], top=10, min_freq=1, 
                                 by="Number of documents", x="Number of documents", y="Total number of citations",
                                 s="H-index", c="Average year of publication", l="CA Country",
                                 max_size=100, x_scale="log", y_scale="log", 
                                 show_mean_line=False, show_mu=False, show_reg_line=False,
                                 arrowprops=None, core=True,                                
                                 kwds1={}, kwds2={}):                      
        if type(items) == str:
            if items.lower() == "eu":
                items = miscbib.eu_countries
        else:
            items = miscbib.select_items(self.ca_countries_df, "CA Country", top=top, min_freq=min_freq, items=items, exclude=exclude)
        df = self.ca_countries_df.loc[self.ca_countries_df["CA Country"].isin(items)]

        f_name = self.res_folder +"\\plots\\top CA countries" if self.save_results else None

        plotbib.scatter_plot_df(df, x, y, s_var=s, c_var=c, l_var=l,
                                 max_size=max_size, cmap=self.cmap, 
                                 d_cmap=self.d_cmap, x_scale=x_scale, y_scale=y_scale,
                                 f_name=f_name,
                                 show_mean_line=show_mean_line, show_mu=show_mu, show_reg_line=show_reg_line,
                                 arrowprops=arrowprops, core=core)

        if self.save_results:
            im = miscbib.BibImage(self.res_folder + "\\plots\\top CA countries", 
                     name="top CA countries")
            self.images.append(im)            
    
    
    def scatter_plot_top_fields(self, items=[], exclude=["nan"], top=10, min_freq=1, 
                                 by="Number of documents", x="Number of documents", y="Total number of citations",
                                 s="H-index", c="Average year of publication", l="Field",
                                 max_size=100, x_scale="log", y_scale="log", max_text_len=30,
                                 show_mean_line=False, show_mu=False, show_reg_line=False,
                                 arrowprops=None, core=True,                                
                                 kwds1={}, kwds2={}):
        items = miscbib.select_items(self.fields_df, "Field", top=top, min_freq=min_freq, items=items, exclude=exclude)
        df = self.fields_df.loc[self.fields_df["Field"].isin(items)]

        f_name = self.res_folder +"\\plots\\top fields" if self.save_results else None

        plotbib.scatter_plot_df(df, x, y, s_var=s, c_var=c, l_var=l,
                                 max_size=max_size, cmap=self.cmap, max_text_len=max_text_len,
                                 d_cmap=self.d_cmap, x_scale=x_scale, y_scale=y_scale,
                                 f_name=f_name,
                                 show_mean_line=show_mean_line, show_mu=show_mu, show_reg_line=show_reg_line,
                                 arrowprops=arrowprops, core=core)

        if self.save_results:
            im = miscbib.BibImage(self.res_folder + "\\plots\\top fields", 
                     name="top fields")
            self.images.append(im)
            
    
    def scatter_plot_top_authors(self, items=[], exclude=["nan"], top=10, min_freq=1, 
                                 by="Number of documents", x="Number of documents", y="Total number of citations",
                                 s="H-index", c="Average year of publication", l="Author",
                                 max_size=100, x_scale="log", y_scale="log", max_text_len=30,
                                 show_mean_line=False, show_mu=False, show_reg_line=False,
                                 arrowprops=None, core=True,                                
                                 kwds1={}, kwds2={}):
        items = miscbib.select_items(self.authors_df, "Author", top=top, min_freq=min_freq, items=items, exclude=exclude)
        df = self.authors_df.loc[self.authors_df["Author"].isin(items)]

        f_name = self.res_folder +"\\plots\\top authors" if self.save_results else None

        plotbib.scatter_plot_df(df, x, y, s_var=s, c_var=c, l_var=l,
                                 max_size=max_size, cmap=self.cmap, max_text_len=max_text_len,
                                 d_cmap=self.d_cmap, x_scale=x_scale, y_scale=y_scale,
                                 f_name=f_name,
                                 show_mean_line=show_mean_line, show_mu=show_mu, show_reg_line=show_reg_line,
                                 arrowprops=arrowprops, core=core, **kwds1)

        if self.save_results:
            im = miscbib.BibImage(self.res_folder + "\\plots\\top authors", 
                     name="top authors")
            self.images.append(im)
            
            
    def scatter_plot_top_keywords(self, items=[], exclude=["nan"], top=10, min_freq=1, 
                                 by="Number of documents", x="Number of documents", y="Total number of citations",
                                 s="H-index", c="Average year of publication", l="Keyword",
                                 max_size=100, x_scale="log", y_scale="log", max_text_len=30,
                                 show_mean_line=False, show_mu=False, show_reg_line=False,
                                 arrowprops=None, core=True,                                
                                 kwds1={}, kwds2={}):
        items = miscbib.select_items(self.keywords_df, "Keyword", top=top, min_freq=min_freq, items=items, exclude=exclude)
        df = self.keywords_df.loc[self.keywords_df["Keyword"].isin(items)]

        f_name = self.res_folder +"\\plots\\top keywords" if self.save_results else None

        plotbib.scatter_plot_df(df, x, y, s_var=s, c_var=c, l_var=l,
                                 max_size=max_size, cmap=self.cmap, max_text_len=max_text_len,
                                 d_cmap=self.d_cmap, x_scale=x_scale, y_scale=y_scale,
                                 f_name=f_name,
                                 show_mean_line=show_mean_line, show_mu=show_mu, show_reg_line=show_reg_line,
                                 arrowprops=arrowprops, core=core, **kwds1)

        if self.save_results:
            im = miscbib.BibImage(self.res_folder + "\\plots\\top keywords", 
                     name="top keywords")
            self.images.append(im)
    
    
    # box-plots (on original data)
    
    def box_plot_top_sources(self, y_var="Cited by", items=[], exclude=["nan"], top=5,
                         boxtype="box", rotate=90, fontsize=10, core=False, **kwds):
                
        d = miscbib.get_cnd_dist(self.df, "Source title", y_var, items=items, 
                                 exclude=exclude, top=top, kind=1)
        im = plotbib.box_plot_from_d(d, boxtype=boxtype, top=top, 
                    rotate=rotate, xlabel="Source title", ylabel=y_var,
                    label_dict=self.dct_sources_abb, save_results=self.save_results,
                    file_name=self.res_folder+f"\\plots\\top sources boxplot {y_var}", core=core,
                    dpi=self.dpi, fontsize=fontsize, **kwds)  
        if im is not None:
            self.images.append(im)
            
    def box_plot_top_authors(self, y_var="Cited by", items=[], exclude=["nan"], top=5,
                         boxtype="box", rotate=90, fontsize=10, core=False, **kwds):
              
        exclude += ["[No author name available]"]
        d = miscbib.get_cnd_dist(self.df, "Authors", y_var, items=items, 
                                 exclude=exclude, top=top, kind=2, sep=", ")
        im = plotbib.box_plot_from_d(d, boxtype=boxtype, top=top, 
                    rotate=rotate, xlabel="Author", ylabel=y_var,
                    save_results=self.save_results,
                    file_name=self.res_folder+f"\\plots\\top authors boxplot {y_var}", core=core,
                    dpi=self.dpi, fontsize=fontsize, **kwds)  
        if im is not None:
            self.images.append(im)
            
    # dynamics plots
    
    def plot_source_dynamics(self, items=[], max_items=5, xlabel="Year", ylabel="Number of documents", 
                  title=None, cum=False, years=None, l_title="Source title", core=False, **kwds):
        c = "cumulative " if cum else ""
        if len(items) == 0 and max_items is not None:
            items = list(self.sources_dyn_df.columns[1:max_items+1])
            items += ["cum " + it for it in items] # tole je malo grdo, bo treba olepšati
        f_name = self.res_folder+r"\\plots\\{c}source dynamics"
        im = plotbib.plot_dynamics(self.sources_dyn_df, items=items, 
                                     xlabel=xlabel, ylabel=ylabel, 
                                     title=title, cum=cum, years=years, 
                                     l_title=l_title,  f_name=f_name, 
                                     save_results=self.save_results, 
                                     core=core, dpi=self.dpi, **kwds)
        if im is not None:
            self.images.append(im)

    # specific plots

    def plot_spectrosopy(self, core=False, **kwds):
        f_name=self.res_folder+"\\plots\\spectroscopy"
        if not hasattr(self, "ref_years_0"):
            pass
        im = plotbib.spectroscopy_plot(self.ref_years_0, f_name=f_name, dpi=self.dpi, 
                                  save_results=self.save_results, core=core, c=self.color_plot, **kwds)
        if im is not None:
            self.images.append(im)
            
    def scatter_plot_ref_age(self, year_range=None):
        df = self.df
        if year_range is not None:
            df = df[df["Year"].between(year_range[0], year_range[1])]
        plotbib.scatter_plot_df(df, "Year", "Average year from refs",
                                 s_var="Number of references", c_var="Cited by", l_var=None,
                                 max_size=100, x_scale="linear", y_scale="linear", 
                                 show_reg_line=True)
        
        
    def scatter_plot_ref_age_density(self, year_range=None, avg=True):
        if avg:
            df = self.df
            x_var, y_var = "Year", "Average year from refs"
        else:
            df = self.df_doc_ref_pairs
            x_var, y_var = "Year", "Year of reference"
        if year_range is not None:
            df = df[df["Year"].between(year_range[0], year_range[1])]
        df = df[[x_var, y_var]].dropna()
        x, y = df[x_var], df[y_var]
        xy = np.vstack([x,y])
        from scipy.stats import gaussian_kde
        z = gaussian_kde(xy)(xy)
        df["Density"] = z
            
        plotbib.scatter_plot_df(df, x_var, y_var,
                                 s_var=None, c_var="Density", l_var=None,
                                 max_size=100, x_scale="linear", y_scale="linear", 
                                 show_reg_line=False, show_id_line=True, cmap=self.cmap)
        
    def scatter_plot_citations_citescore(self, x="CiteScore", y="Cited by",
                                         s=None, c="Year", l=None, 
                                         show_id_line=True):
        vrs = [v for v in [x,y,s,c,l] if v is not None]
        df = self.df[vrs].dropna()
        plotbib.scatter_plot_df(df, x, y, s_var=s, c_var=c, l_var=l,
                                 max_size=100, x_scale="linear", y_scale="linear", 
                                 show_reg_line=False, show_id_line=True, cmap=self.cmap)
        
        
    def scatter_plot_mds(self, x=0, y=1, f_name="mds plot"):
        if f_name is not None:
            f_name=self.res_folder+"\\plots\\" + f_name
        plotbib.scatter_plot_df(self.df_mds, f"MDS {x}", f"MDS {y}", l_var="label", 
                                c_var="group", x_scale="linear", y_scale="linear", 
                                axis_off=True, f_name=f_name)
        
   
        
    def trend_topics(self, what="ak", items=[], exclude=[], name_var=None, period=None, 
                     left_var="Q1 year", right_var="Q3 year", mid_var="median year", 
                     s_var="Number of documents", c_var="Total number of citations",
                     items_per_year=5, x_label="Year (Q1-median-Q3)"):
        
        if what in ["ak", "ik", "aik"]:
            df_prod, l_var, f_name = self.keywords_df, "Keyword", self.res_folder+"\\plots\\trend topics keywords {what}"
        elif what.lower() == "sources":
            df_prod, l_var, f_name = self.sources_df, "Source title", self.res_folder+"\\plots\\trend sources"
        elif what.lower() == "countries":
            df_prod, l_var, f_name = self.ca_countries_df, "CA Country", self.res_folder+"\\plots\\trend countries"
        elif what.lower() == "authors":
            df_prod, l_var, f_name = self.authors_df, "Author", self.res_folder+"\\plots\\trend authors"
        elif what.lower() in ["abstract", "title"]:
            df_prod, l_var, f_name = self.words_and_phrases_df, "word", self.res_folder+f"\\plots\\trend topics {what}"
        
        
        df_prod = miscbib.prepare_for_trend_plot(df_prod, items=items, exclude=exclude, name_var=l_var,
                                   mid_var=mid_var, period=period, items_per_year=items_per_year)
       
        im = plotbib.trend_plot(df_prod, left_var=left_var, right_var=right_var, mid_var=mid_var,
                       s_var=s_var, c_var=c_var,
                       main_color="blue", x_label=x_label,
                       l_var=l_var, max_size=100, cmap=self.cmap,
                       f_name=f_name, dpi=self.dpi, save_results=self.save_results, core=False)
        if im is not None:
            self.images.append(im)
            
    def plot_country_collab_production(self, s_var="SCP", m_var="MCP",
                                l_var="Country of corresponding author", top=10,
                                f_name="country collaboration production", core=False, **kwds):
        self.country_collab_df = self.country_collab_df[self.country_collab_df["Country of corresponding author"]!="nan"]
        im = plotbib.plot_country_collab_production(self.country_collab_df, s_var= s_var, m_var=m_var,
                                    l_var=l_var, top=top,
                                    f_name=self.res_folder+"\\plots\\country collaboration", dpi=self.dpi,
                                    save_results=self.save_results, core=core, **kwds)
        if im is not None:
            self.images.append(im)
            
    def plot_word_cloud(self, by="abstracts", name_var=None, freq_var=None, color_var="Average year of publication",
                              background_color="white", f_name="wordcloud", dpi=600, **kwds):
        
        if by == "abstracts":
            perf_df = self.words_and_phrases_df
            if name_var is None:
                name_var = "word"
            if freq_var is None:
                freq_var = "f"
        elif by == "keywords":
            perf_df = self.keywords_df
            if name_var is None:
                name_var = "Keyword"
            if freq_var is None:
                freq_var = "Number of documents"
        if f_name is not None:
            f_name=self.res_folder+"\\plots\\" + f_name
        plotbib.plot_word_cloud_from_freqs(perf_df, name_var=name_var, freq_var=freq_var,
                                           color_var=color_var,background_color=background_color,
                                           f_name=f_name, dpi=self.dpi)
        if self.save_results:
            im = miscbib.BibImage(f_name, name="wordcloud")
            self.images.append(im)
        
    def plot_loadings(self, c1=0, c2=1, min_norm=0.1, col1="r", col2="b", 
                      l1=None, l2=None, f_name="loadings", show_grid=True):
        
        a = self.decomposition_method + "_loadings_" 
        a1, a2 = a + "X" + "_df", a + "Y" + "_df"
        df_l1, df_l2 = getattr(self.two_aspects, a1), getattr(self.two_aspects, a2)
        if f_name is not None:
            f_name = self.res_folder + "\\plots\\" + f_name + " " + self.decomposition_method

        l1 = self.rel1 if l1 is None else None
        l2 = self.rel2 if l2 is None else None
        
        plotbib.loadings_plot(df_l1, df_l2, c1=c1, c2=c2, min_norm=min_norm, col1=col1, col2=col2, 
                              l1=l1, l2=l2, show_grid=show_grid, f_name=f_name, dpi=self.dpi)
        
    def k_fields_plot(self, fields=["keywords", "sources"], customs=[], ks=10, 
                      color="Average year", add_colorbar=False, save_html=False, font_size=10):
        # custom mora biti par (ime, dataframe), mora pa biti custom 1, custom 2 ... v fileds na pravem mestu
        # če hočeš specificirati, katera polja se vzame, potem to vnaprej pripravi s klici funkcij
        # če uporabljaš customs, sam napiši, kaj naj bodo ks
        
        gen_fields = {"keywords": ("keywords_df_ind", "get_keywords_stats"),
                      "sources": ("sources_df_ind", "get_sources_stats"),
                      "authors": ("authors_df_ind", "get_authors_stats"),
                      "ca countries": ("ca_countries_df_ind", "get_ca_countries_stats")}
        dfs, ks_, names = [], {}, []
        for i, field in enumerate(fields):
            if "custom" in field:
                j = [int(s) for s in field.split() if s.isdigit()][0] - 1
                name, df_ = customs[j]
                print(name, df_)
                dfs.append(df_)
                ks_[j] = len(df_.columns)
                names.append(name)
            else:
                df, f = gen_fields[field]
                if not hasattr(self, df):
                    if isinstance(ks, int):
                        getattr(self, f)(top=ks)
                    elif isinstance(ks, list):
                        getattr(self, f)(top=ks[i])
                dfs.append(getattr(self, df))
                names.append(field)
        
        if color == "Average year":
            add_ser, fn = self.df["Year"], np.mean
        elif color == "Citations per document":
            add_ser, fn = self.df["Cited by"], np.mean
        elif color == "H-index":
            add_ser, fn = self.df["Cited by"], miscbib.h_index
        else:
            add_ser, fn = None, np.mean
            
       
        dicts = []
        if "authors" in fields:
            dicts.append(self.dct_au)
        if "sources" in fields:
            dicts.append(self.dct_sources_abb)
            
            
        self.pairs_sankey, labels, color_vals, ks, groups = utilsbib.prepare_for_sankey(
            dfs, ks=ks, dicts=dicts,
            add_ser=add_ser, fn=fn)
        

        k = len(ks)
                
        if self.save_results:
            f_name = self.res_folder +"\\plots\\%d-fields-plot" % k
        else:
            f_name = None
        
        plotbib.plot_sankey(self.pairs_sankey, labels, color_vals, ks, font_size=font_size,
                             f_name=f_name, c_names=names, 
                             cmap=self.cmap, single_color=self.color_plot,
                             save_html=save_html)
        
    def plot_tree_map(self, field="keywords", size_var="Number of documents", color_var="Average year of publication",
                 labels_var=None, show=None, font_size=10):
        
        try:
            df_perf = getattr(self, field + "_df")
            if labels_var is None:
                labels_var=df_perf.columns[0]
        except:
            print("field not valid")
            return None
        
        f_name = self.res_folder + "\\plots\\treemap of " + field 
        
        plotbib.plot_treemap(df_perf, size_var=size_var, color_var=color_var,
                           labels_var=labels_var, show=show, cmap_name=self.cmap, 
                           ndig=self.sig_digits, font_size=font_size, f_name=f_name, dpi=self.dpi)
        
        
    def plot_lotka_law(self, what="authors", f_name="lotka", **kwds):
        
        if f_name is not None:
            f_name += " "
            f_name += what
        
        if what == "authors":
            if not hasattr(self, "authors_f2_df"):
                print("Author's counts not computed")
                return None
            n_values, f_values = self.authors_f2_df["Number of documents"], self.authors_f2_df["Number of authors"]
            results = utilsbib.compute_lotka_law(n_values, f_values)
            if f_name is not None:
                f_name = self.res_folder + "\\plots\\" + f_name
        elif what == "sources":
            if not hasattr(self, "sources_f2_df"):
                print("Sources' counts not computed")
                return None
            n_values, f_values = self.sources_f2_df["Number of documents"], self.sources_f2_df["Number of sources"]
            results = utilsbib.compute_lotka_law(n_values, f_values)
            if f_name is not None:
                f_name = self.res_folder + "\\plots\\" + f_name
        elif what in ["countries", "ca countries"]:
            if not hasattr(self, "ca_countries_f2_df"):
                print("Countries counts not computed")
                return None
            n_values, f_values = self.ca_countries_f2_df["Number of documents"], self.ca_countries_f2_df["Number of countries"]
            results = utilsbib.compute_lotka_law(n_values, f_values)
            if f_name is not None:
                f_name = self.res_folder + "\\plots\\" + f_name
            
        
        plotbib.plot_lotka_law(n_values, f_values, results, ylabel=f"Number of {what}", f_name=f_name, dpi=self.dpi, **kwds)
    
        
    def basic_analysis(self, methods=None):
        
        if methods is None:
            methods = [self.get_main_info, self.get_production, self.plot_production, self.get_top_cited_docs,
                      self.count_sources, self.count_ca_countries, self.count_authors,
                      self.count_keywords, self.get_sources_stats, self.get_ca_countries_stats,
                      self.get_authors_stats, self.scatter_plot_top_sources,
                      self.scatter_plot_top_ca_countries, 
                      self.get_keyword_co_net, self.get_keyword_co_net, self.to_excel]
        
        for m in methods:
            try:
                m()
            except:
                print("Problem with running ", m)
        