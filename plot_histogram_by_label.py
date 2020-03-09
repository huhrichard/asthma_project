import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter



# pollutant_name = 'NH4y0to2'
pollutant_name = 'SO4y-1'


# asthma_df = pd.read_csv('data/exposures-4yrs-filtered_race_in1Col.csv')
# asthma_df = pd.read_csv('data/exposures-4yrs-filtered_0.8_race_in1Col_svdImpute.csv')
asthma_df = pd.read_csv('data/exposure_7pollutants_no_impute_y-1.csv')

asthma_df = asthma_df[[pollutant_name, 'label']]

so4yr4_asthma = asthma_df[asthma_df['label']==1][pollutant_name]
so4yr4_no_asthma = asthma_df[asthma_df['label']==0][pollutant_name]

hwa, bwa = np.histogram(so4yr4_asthma, bins=len(list(set(so4yr4_asthma))))
hwoa, bwoa = np.histogram(so4yr4_no_asthma, bins=len(list(set(so4yr4_no_asthma))))

# pollutant_name += '_imputed'
fig = plt.figure()
ax = fig.add_subplot(111)
width = 0.
# ax.bar([bwa, bwoa], [hwa.astype(float)/hwa.sum(), hwoa.astype(float)/hwoa.sum()] label=['with asthma', 'w/o asthma'])
# ax.bar([bwa, bwoa], [hwa.astype(float)/hwa.sum(), hwoa.astype(float)/hwoa.sum()] label=['with asthma', 'w/o asthma'])
ax.hist([so4yr4_asthma, so4yr4_no_asthma],
        weights=[np.ones(len(so4yr4_asthma))/len(so4yr4_asthma), np.ones(len(so4yr4_no_asthma))/len(so4yr4_no_asthma)],
        label=['with asthma', 'w/o asthma'])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel('pollutant level')
ax.set_ylabel('percentage of pollutant level (by category)')
ax.legend(loc='upper right')
# ax.set_title(pollutant_name+'_imputed_data')
pollutant_list = pollutant_name.split('y')
def year_str_for_title(yr):
    if yr == '0':
        return ''
    elif int(yr) > 0:
        return '+' + yr
    else:
        return yr
if 'to' in pollutant_list[-1]:
    pollutant_start_yr, pollutant_end_yr = pollutant_list[-1].split('to')

    title_start_yr = year_str_for_title(pollutant_start_yr)
    title_end_yr = year_str_for_title(pollutant_end_yr)

    ax.set_title('Histogram of {} level from (birth year{}) to (birth year{})'.format(pollutant_list[0], title_start_yr, title_end_yr))
else:
    pollutant_yr = year_str_for_title(pollutant_list[-1])

    ax.set_title('Histogram of {} level at (birth year{})'.format(pollutant_list[0], pollutant_yr))

plt.show()

fig.savefig("{}_histogram.png".format(pollutant_name))