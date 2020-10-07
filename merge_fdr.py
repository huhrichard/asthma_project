import sys
import os
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

suffix = sys.argv[-1]
outcome_binary_dict = {
    'asthma': True,
    # 'asthma(act_score)':False,
    'age_greaterthan5_diagnosed_asthma': True,
    # 'age_diagnosed_asthma': False,
    'daily_controller_past6months': True,
    'emergency_dept': True,
    # 'emergency_dept_pastyr_count': False,
    'hospitalize_overnight': True,
    # 'hospitalize_overnight_pastyr_count': False,
    # 'regular_asthma_symptoms_past6months': True,
    # 'regular_asthma_symptoms_daysCount_pastWeek': False
    #  'asthma(act_score_lessthan20)': True,
    #  'emergency_dept_pastyr_count(greaterthan0)': True,
    #  'emergency_dept_pastyr_count(greaterthan_nz_median)': True,
    #  'hospitalize_overnight_pastyr_count(greaterthan0)': True,
    #  'hospitalize_overnight_pastyr_count(greaterthan_nz_median)': True,
    #  'regular_asthma_symptoms_daysCount_pastWeek(greaterthan0)': True,
    'regular_asthma_symptoms_daysCount_pastWeek(greaterthan_nz_median)': True
}
p_val_list = []
pred_score_list = []
for outcome in outcome_binary_dict:
    p_val_list.append(pd.read_csv('fdr_{}_{}.csv'.format(outcome, suffix)))
    pred_score_list.append(pd.read_csv('pred_score_{}_{}.csv'.format(outcome, suffix)))

pvalue_df_cat = pd.concat(p_val_list, ignore_index=True)
pred_score_df_cat = pd.concat(pred_score_list, ignore_index=True)

pvalue_df_cat['fdr'] = 0.0
bool_list_fdr = [
                pvalue_df_cat['outcome']=='asthma',
                 (pvalue_df_cat['outcome']!='asthma') & (pvalue_df_cat['binary_outcome'] == True),
                 (pvalue_df_cat['outcome'] != 'asthma') & (pvalue_df_cat['binary_outcome'] == False)
                 ]
for b in bool_list_fdr:
    pvalue_df_cat.loc[b,'fdr'] = fdrcorrection(pvalue_df_cat.loc[b,'p_val'].values)[-1]

pvalue_df_cat.to_csv('fdr_{}.csv'.format(suffix), index=False, header=True)
pred_score_df_cat.to_csv('pred_score_{}.csv'.format(suffix), index=False, header=True)

