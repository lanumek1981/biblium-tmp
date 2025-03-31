# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 23:11:43 2023

@author: Lan
"""

from bibgroup import BiblioGroup
import predictbib

import pandas as pd

class PredictBiblioGroup(BiblioGroup):
    
    def set_x(self, predictors="keywords", items=[], exclude=["nan"], min_freq=5, top=20, override=True):
        if predictors == "keywords":
            if (not hasattr(self, "keywords_df_ind")) or override:
                self.ba.count_keywords()
                self.ba.get_keywords_stats(items=items, exclude=exclude, top=top)
                
                column_sums = self.ba.keywords_df_ind.sum()
                filtered_columns = column_sums[column_sums >= min_freq].index
                self.ba.keywords_df_ind = self.ba.keywords_df_ind[filtered_columns]
                
                
            return self.ba.keywords_df_ind
    
    def log_reg_stat(self, X=None, predictors="keywords", min_freq=5, items=[], exclude=["nan"], file_name=None, alpha=0.05, 
                     remove_constant_cases=True, missings_ind=None, top=20, **kwds):
        if X is None:
            X = self.set_x(predictors=predictors, items=items, exclude=exclude, min_freq=min_freq, top=top, **kwds)
        
        if missings_ind is not None:
            X = X[~missings_ind]
        elif remove_constant_cases:
            missings_ind = self.ba.df[predictors].isna()
            missings_ind = missings_ind.reindex(X.index, fill_value=False)
            X = X[~missings_ind]

        self.lr_models = {}
        writer = pd.ExcelWriter(file_name) if file_name else None

        summary_data = []

        for c in self.g_ind_df.columns:
            y = self.g_ind_df[c]
            if missings_ind is not None:
                y = [y[i] for i in range(len(missings_ind)) if missings_ind[i] == 0]

            lr = predictbib.log_reg(X, y)
            lr_summary = lr.summary2()
            self.lr_models[c] = lr
            
            summ, table = [pd.DataFrame(t) for t in lr_summary.tables]
            table = table.reset_index()

            # Apply conditional formatting to 'P>|z|' column
            styled_table = table.style.applymap(lambda val: f'background-color: {"lightgreen" if float(val) < alpha else "white"}', subset=['P>|z|'])

            c_name = c if "=" not in c else c.split("=")[1]
            while " " in c_name:
                c_name = c_name.replace(" ", "_")
            setattr(self, "lr_table_" + c_name + "_df", table)
            setattr(self, "lr_summary_" + c_name + "_df", summ)
            
            coef_pz = table[['index', 'Coef.', 'P>|z|']]
            coef_pz.columns = ['index', f'{c}_Coef.', f'{c}_P>|z|']
            summary_data.append(coef_pz)

            # Save to Excel file
            if writer:
                if len(c_name) > 19:
                    c_name = c_name[:19]
                styled_table.to_excel(writer, sheet_name=c_name + "_lr_table", index=False)
                summ.to_excel(writer, sheet_name=c_name + "_lr_summary", index=False)

        combined_summary = pd.concat(summary_data, axis=1)

        # Keep only the first occurrence of the 'index' column
        combined_summary = combined_summary.loc[:, ~combined_summary.columns.duplicated()]

        # Save the combined summary to a new sheet
        if writer:
            combined_summary.to_excel(writer, sheet_name='combined_summary', index=False)
            
        # Save and close Excel file
        if writer:
            writer.save()
            writer.close()

    
    
    
    def classify(self, X=None, predictors="keywords", items=[], exclude=[], 
                          top=20, model="lda", test_size=0.2,
                 metrics=["accuracy", "auc", "sensitivity", "precision", "recall"], **kwds):
        if X is None:
            X = self.set_x(predictors=predictors, items=items, exclude=exclude, top=top)
        for c in self.g_ind_df.columns:
            y = self.g_ind_df[c]
            Y_pred, clf, evaluation_results = predictbib.predict_y_from_x(
                X, y, test_size=0.2, model=model, metrics=metrics, **kwds)
            c_name = c if "=" not in c else c.split("=")[1]
            while " " in c_name:
                c_name = c_name.replace(" ", "_")
            setattr(self, model + "_" + c_name + "_classifier", clf)
            setattr(self, model + "_" + c_name + "_evals", evaluation_results)
            
            evals_df = pd.DataFrame.from_dict(evaluation_results, orient="index", columns=["value"])
            
            setattr(self, model + "_" + c_name + "_evals_df", evals_df)
            