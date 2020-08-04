import matplotlib
import matplotlib.pyplot as plt
import os, fnmatch
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13

# plt.rc('xtick', labelsize=BIGGER_SIZE)
# plt.rc('ytick', labelsize=BIGGER_SIZE)
# plt.rc('legend', labelsize=BIGGER_SIZE)
plt.rcParams.update({'font.size': BIGGER_SIZE})
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def plot_histogram(asthma_df, plot_dir, pollutant_name, label_plot=['with asthma', 'w/o asthma']):
    # pollutant_col_name = asthma_df.columns
    df = asthma_df[[pollutant_name, 'label']]

    # pollutant_level_w_asthma = np.log10(df[asthma_df['label'] == 1][pollutant_name])
    # pollutant_level_wo_asthma = np.log10(df[asthma_df['label'] == 0][pollutant_name])
    pollutant_level_w_asthma = (df[asthma_df['label'] == 1][pollutant_name].values)
    pollutant_level_wo_asthma = (df[asthma_df['label'] == 0][pollutant_name].values)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, bins, _ = ax.hist([pollutant_level_w_asthma, pollutant_level_wo_asthma],
            weights=[np.ones(len(pollutant_level_w_asthma))/len(pollutant_level_w_asthma), np.ones(len(pollutant_level_wo_asthma))/len(pollutant_level_wo_asthma)],
            label=label_plot,
            color=['red', 'green'])



    ax.legend(loc='upper right')
    # ax.axvline(float(thres), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    min_xlim, max_xlim = ax.get_xlim()
    y_pos = 0.75
    # ax.text(float(thres)+(max_xlim-min_xlim)*0.02, (max_ylim*y_pos+(1-y_pos)*min_ylim), 'Threshold = {:.3e}'.format(thres))
    # ax.set_title(pollutant_name+'_imputed_data')

    # pollutant_list = pollutant_name.split('y')
    # ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('{} level ({} scale)'.format(subscript_like_chemical_name(pollutant_list[0]), '${log_{10}}$'))
    ax.set_xlabel('{} level'.format(pollutant_name))
    ax.set_ylabel('Fraction of children exposed (separated by class)')
    # y_ticks = ax.get_yticks
    fig.savefig(os.path.join(plot_dir, "histogram_{}.png".format(pollutant_name)), bbox_inches="tight")

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, spacing='proportional',  **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0,
             # ha="right",
             rotation_mode="anchor")


    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_aspect(aspect='auto', adjustable='box')
    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
    #
            # text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
    #         texts.append(text)

    return texts

plot_dir = "./plot/"
# fdr_path = './plot/nata_diagnose_year/fdr.csv'
# fdr_path = './plot/nata_birth_year/fdr.csv'


# outcome_list = ['act_score',
#                 'age_greaterthan5_diagnosed_asthma',
#                 # 'age_di'
#                 'daily_controller_past6months',
#                 'emergency_dept',
#                 'hospitalize_overnight',
#                 'regular_asthma_symptoms_past6months']
outcome_list = [
                'asthma',
                'asthma(act_score)',
                'age_greaterthan5_diagnosed_asthma',
                'age_diagnosed_asthma',
                'daily_controller_past6months',
                'emergency_dept',
                'emergency_dept_pastyr_count',
                'hospitalize_overnight',
                'hospitalize_overnight_pastyr_count',
                'regular_asthma_symptoms_past6months',
               'regular_asthma_symptoms_daysCount_pastWeek'
                ]


outcome_col_rename = {
                        'asthma':'asthma_(Binary)',
                        'asthma(act_score)': 'act_score_(Cont.)',
                        'age_greaterthan5_diagnosed_asthma': 'age>5_diagnosed_asthma_(Binary)',
                        'age_diagnosed_asthma': 'age_diagnosed_asthma_(Cont.)',
                        'daily_controller_past6months': 'daily_controller_past6months_(Binary)',
                        'emergency_dept': 'emergency_dept_(Binary)',
                        'emergency_dept_pastyr_count': 'emergency_dept_pastyr_(Cont.)',
                        'hospitalize_overnight': 'hospitalize_overnight_(Binary)',
                        'hospitalize_overnight_pastyr_count': 'hospitalize_overnight_pastyr_(Cont.)',
                        'regular_asthma_symptoms_past6months': 'regular_asthma_symptoms_(Binary)',
                       'regular_asthma_symptoms_daysCount_pastWeek': 'regular_asthma_symptoms_(Cont.)'
                           }

# outcome_col_rename = {'act_score':'ACT',
#                 'age_greaterthan5_diagnosed_asthma':'age>5',
#                 'daily_controller_past6months': 'daily_controller',
#                 'emergency_dept':'emerg_dept',
#                 'hospitalize_overnight':'hos_night',
#                 'regular_asthma_symptoms_past6months':'reg_symptoms'}

outcome_reverse_dict = {v: k for k, v in outcome_col_rename.items()}

# pollutant_list = ['EC', 'OC', 'SO4', 'NH4', 'Nit', 'NO2', 'PM2.5']
relation_dict = {'pos_correlate': 2.0,
                 # 'mixed_sign_profile': 0.0,
                 'neg_correlate': 1.0,
                 # 'na': 0
                 }
relation_list = [k for k in relation_dict]
# relation_list = ['NA'] + relation_list
# relation_list.append('NA')
relation_list.append('NA')

relation_inv_dict = {v: k for k, v in relation_dict.items()}



# p_val_cont = fdr_df.loc[fdr_df.apply(lambda x: '(Cont.)' in x['outcome'], axis=1), 'p_val'].values
# p_val_bin = fdr_df.loc[fdr_df.apply(lambda x: '(Binary)' in x['outcome'], axis=1), 'p_val'].values
#
# _, fdr_cont = fdrcorrection(p_val_cont)
# _, fdr_bin = fdrcorrection(p_val_bin)
#
# fdr_df.loc[fdr_df.apply(lambda x: '(Cont.)' in x['outcome'], axis=1), 'fdr'] = fdr_cont
# fdr_df.loc[fdr_df.apply(lambda x: '(Binary)' in x['outcome'], axis=1), 'fdr'] = fdr_bin



def summarize_plot(col='p_val', pollutant_suffix=''):
    print(pollutant_suffix)
    # for idx, outcome in enumerate(outcome_list):
    fdr_path = './fdr.csv'
    fdr_df = pd.read_csv(fdr_path, sep=',')
    # print(fdr_df)
    str_matched = fdr_df['profile'].str.contains(pollutant_suffix, regex=False)
    # print(str_matched)
    fdr_df = fdr_df[str_matched]
    fdr_df = fdr_df.loc[fdr_df['fdr'] < 0.05]
    fdr_df['profile'] = fdr_df['profile'].str.replace(pollutant_suffix, '', regex=False)
    single_pollutantTimeWin = dict()
    multi_pollutantTimeWin = dict()
    single_pollutant_fdr = dict()
    multi_pollutant_fdr = dict()
    single_pollutant_coef = dict()
    multi_pollutant_coef = dict()
    for old_name, new_name in outcome_col_rename.items():
        fdr_df.replace(old_name, new_name, inplace=True)
    # print(fdr_df)
    new_outcome_list = [outcome_col_rename[old_name] for old_name in outcome_list]
    outcome_idx_dict = {out: idx for idx, out in enumerate(new_outcome_list)}

    for row_idx, pollutant_row in fdr_df.iterrows():
        idx = outcome_idx_dict[pollutant_row['outcome']]
        pollutant_profile = pollutant_row['profile'].split('\t\t')

        p_wo_thres = []
        relation_to_outcome = pollutant_row['relation']
        # print(pollutant_row)
        if len(pollutant_profile) == 1:
            set_p_fdr = pollutant_profile[0].split('>')[0].split('<')[0]
            if set_p_fdr in single_pollutant_fdr:
                single_pollutant_fdr[set_p_fdr][idx] = pollutant_row[col]
                single_pollutant_coef[set_p_fdr][idx] = abs(pollutant_row['coef'])
            else:
                single_pollutant_fdr[set_p_fdr] = np.ones(len(outcome_list))
                single_pollutant_fdr[set_p_fdr][idx] = pollutant_row[col]
                single_pollutant_coef[set_p_fdr] = np.zeros(len(outcome_list))
                single_pollutant_coef[set_p_fdr][idx] = abs(pollutant_row['coef'])
        else:
            print(pollutant_profile)
            for p in pollutant_profile:
                if '>' in p:
                    p_wo_thres.append(p.split('>')[0] + '>')
                else:
                    p_wo_thres.append(p.split('<=')[0] + '<=')
            set_p_fdr = frozenset(p_wo_thres)
            if set_p_fdr in multi_pollutant_fdr:
                multi_pollutant_fdr[set_p_fdr][idx] = pollutant_row[col]
                multi_pollutant_coef[set_p_fdr][idx] = abs(pollutant_row['coef'])
            else:
                multi_pollutant_fdr[set_p_fdr] = np.ones(len(outcome_list))
                multi_pollutant_fdr[set_p_fdr][idx] = pollutant_row[col]
                multi_pollutant_coef[set_p_fdr] = np.zeros(len(outcome_list))
                multi_pollutant_coef[set_p_fdr][idx] = abs(pollutant_row['coef'])

        if pollutant_row[col] < 0.05:
            if relation_to_outcome == 'mixed_sign_profile' or pollutant_row[col] > 0.05:
                rel = 0.5
            else:
                rel = relation_dict[relation_to_outcome]

            if len(pollutant_profile) == 1:
                set_p = pollutant_profile[0].split('>')[0].split('<')[0]
                if set_p in single_pollutantTimeWin:
                    single_pollutantTimeWin[set_p][idx] = rel
                else:
                    single_pollutantTimeWin[set_p] = 0.5*np.ones(len(outcome_list))
                    single_pollutantTimeWin[set_p][idx] = rel
            else:
                print(pollutant_profile)
                for p in pollutant_profile:
                    if '>' in p:
                        p_wo_thres.append(p.split('>')[0]+'>')
                    else:
                        p_wo_thres.append(p.split('<=')[0] + '<=')
                set_p = frozenset(p_wo_thres)
                if pollutant_row['coef'] > 0:
                    rel = 2.0
                else:
                    rel = 1.0

                if pollutant_row[col] > 0.05:
                    rel = 0.5
                if set_p in multi_pollutantTimeWin:
                    multi_pollutantTimeWin[set_p][idx] = rel
                else:
                    multi_pollutantTimeWin[set_p] = 0.5*np.ones(len(outcome_list))
                    multi_pollutantTimeWin[set_p][idx] = rel

    m_keys = multi_pollutantTimeWin.copy().keys()
    if len(m_keys)>0:
        for key in m_keys:
            new_key = ''
            k_list = list(key)
            k_list.sort()
            for k in k_list:
                new_key += '{}\n'.format(k)
            new_key = new_key[:-1]
            multi_pollutantTimeWin[new_key] = multi_pollutantTimeWin.pop(key)

    m_keys = multi_pollutant_fdr.copy().keys()
    if len(m_keys) > 0:
        for key in m_keys:
            new_key = ''
            k_list = list(key)
            k_list.sort()
            for k in k_list:
                new_key += '{}\n'.format(k)
            new_key = new_key[:-1]
            multi_pollutant_fdr[new_key] = multi_pollutant_fdr.pop(key)

    m_keys = multi_pollutant_coef.copy().keys()
    if len(m_keys) > 0:
        for key in m_keys:
            new_key = ''
            k_list = list(key)
            k_list.sort()
            for k in k_list:
                new_key += '{}\n'.format(k)
            new_key = new_key[:-1]
            multi_pollutant_coef[new_key] = multi_pollutant_coef.pop(key)

    single_df = pd.DataFrame.from_dict(single_pollutantTimeWin,
                                       orient='index',
                                       columns=new_outcome_list).sort_index()
    multi_df = pd.DataFrame.from_dict(multi_pollutantTimeWin,
                                      orient='index',
                                      columns=new_outcome_list).sort_index()
    single_fdr_df = pd.DataFrame.from_dict(single_pollutant_fdr,
                                       orient='index',
                                       columns=new_outcome_list).sort_index()
    multi_fdr_df = pd.DataFrame.from_dict(multi_pollutant_fdr,
                                      orient='index',
                                      columns=new_outcome_list).sort_index()

    single_coef_df = pd.DataFrame.from_dict(single_pollutant_coef,
                                           orient='index',
                                           columns=new_outcome_list).sort_index()
    multi_coef_df = pd.DataFrame.from_dict(multi_pollutant_coef,
                                          orient='index',
                                          columns=new_outcome_list).sort_index()
    # print(single_df)
    # print(single_df)
    # print(multi_df)
    merged_df = pd.concat([single_df, multi_df])
    merged_fdr_df = pd.concat([single_fdr_df, multi_fdr_df])
    merged_coef_df = pd.concat([single_coef_df, multi_coef_df])

    # merged_df.rename(columns=outcome_col_rename, inplace=True)

    fontsize_pt = plt.rcParams['xtick.labelsize']
    dpi = 72.27

    # comput the matrix height in points and inches
    # matrix_height_pt = fontsize_pt * merged_df.shape[1]
    # print(matrix_height_pt)
    # matrix_height_in = matrix_height_pt / dpi

    # compute the required figure height
    top_margin = 0.04  # in percentage of the figure height
    bottom_margin = 0.04 # in percentage of the figure height
    # figure_height = matrix_height_in / (1 - top_margin - bottom_margin)
    # plt.clf()
    def fig_gen():
        fig = plt.figure(figsize=(25,15))
        ax = fig.add_subplot(111)
        qrates = np.array(relation_list)
        np_array = [0.5-1e-4, 0.5+1e-4, 1.5, 2.5]
        norm = matplotlib.colors.BoundaryNorm(np_array, 3)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

        im, _ = heatmap(merged_df.values, merged_df.index, [c.replace('_', '\n') for c in merged_df.columns], ax=ax,
                        cmap=matplotlib.colors.ListedColormap(['white', 'red', 'green']),
                        norm=norm,
                        cbar_kw=dict(ticks=np.arange(1, 3), format=fmt), cbarlabel="Relation between Profile and Outcome")
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=20)

        return fig, ax, im
    # output file

    fig1, ax1, im1 = fig_gen()
    fig2, ax2, im2 = fig_gen()
    fig1.suptitle('{} Significant Pollutant Profile ({} < 0.05)'.format(pollutant_suffix, col), fontsize=20)
    fig2.suptitle('Coefficients of {} Significant Pollutant Profile'.format(pollutant_suffix), fontsize=20)
    print(merged_coef_df)
    convert_str = {c: str for c in merged_df.columns}
    merged_df = merged_df.astype(convert_str)
    for k, v in relation_dict.items():
        merged_df = merged_df.replace(str(v), k)
    merged_df.replace('0.5', '>0.05', inplace=True)
    yr_dir = pollutant_suffix.replace('+5', '_yr_5').replace('-5', '_yr_-5')
    plot_dir = './plot/nata_{}/histogram_profile_from_{}/'.format(yr_dir, pollutant_suffix)
    # if 'birth' in pollutant_suffix:
    #     plot_dir = plot_dir.format('birth')
    # else:
    #     plot_dir = plot_dir.format('diagnose')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    for idx_r, r in enumerate(merged_df.index):
        for idx_c, c in enumerate(merged_df.columns):
            if merged_fdr_df.loc[r, c] < 0.05:
                plot_txt_fdr = '{:.3e}'.format(merged_fdr_df.loc[r, c])
                plot_txt_coef = '{:.3f}'.format(merged_coef_df.loc[r, c])
                im1.axes.text(idx_c, idx_r, plot_txt_fdr, ha="center", va="center", color="w", size=14, fontweight='bold')
                im2.axes.text(idx_c, idx_r, plot_txt_coef, ha="center", va="center", color="w", size=14, fontweight='bold')
        # if '\n' in r:
        r_list = r.split('\n')
        for sub_r in r_list:
            sub_r = sub_r.replace('<=', '').replace('>', '')
            plot_histogram(asthma_df=asthma_df, plot_dir=plot_dir, pollutant_name=sub_r+'birth+5')

    plt.autoscale()
    fig1.savefig('heatmp_summary_nata_{}_{}.png'.format(pollutant_suffix,col), bbox_inches="tight")
    fig2.savefig('heatmp_summary_nata_{}_{}.png'.format(pollutant_suffix,'coef'), bbox_inches="tight")

    merged_fdr_df.to_csv('summary_{}_{}.csv'.format(col, pollutant_suffix))

# sig_list = ['p_val', 'fdr']
sig_list = ['fdr']
suffix_list = ['birth+5']
asthma_csv_path = 'data/asthma_NATA_birth_yr_5.csv'
asthma_df = pd.read_csv(asthma_csv_path)
for sig in sig_list:
    for suffix in suffix_list:
        summarize_plot(sig, pollutant_suffix=suffix)
# summarize_plot('p_val')
# summarize_plot('fdr')







