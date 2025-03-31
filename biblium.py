# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:29:44 2022

@author: Lan.Umek
"""

from bibstats import BiblioStats
from bibplot import BiblioPlot
from bibnetwork import BiblioNetwork
from bibnetworkplot import PlotBiblioNetwork
from bibgroup import BiblioGroup
from bibgroupplot import PlotBiblioGroup
from bibpredict import PredictBiblioGroup
from bibproto import ProtoBib, ProtoBibGroup

# instalirati je treba sqaurify, wordcloud, adjustText, venn, python-docx in cdlib


class BiblioAnalysis(PlotBiblioNetwork, BiblioPlot):

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
            #self.to_latex()
            
class BiblioGroupAnalysis(PredictBiblioGroup, PlotBiblioGroup):
    
    pass
    
    