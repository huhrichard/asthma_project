import matplotlib
import matplotlib.pyplot as plt
import os, fnmatch
import numpy as np
import pandas as pd

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

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
    ax.set_aspect('auto', 'box')
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
outcome_list = ['act_score',
                'age_greaterthan5_diagnosed_asthma',
                'daily_controller_past6months',
                'emergency_dept',
                'hospitalize_overnight',
                'regular_asthma_symptoms_past6months']
outcome_col_rename = {'act_score':'ACT',
                'age_greaterthan5_diagnosed_asthma':'age>5',
                'daily_controller_past6months': 'daily_controller',
                'emergency_dept':'emerg_dept',
                'hospitalize_overnight':'hos_night',
                'regular_asthma_symptoms_past6months':'reg_symptoms'}


single_pollutantTimeWin = dict()
multi_pollutantTimeWin = dict()
pollutant_list = ['EC', 'OC', 'SO4', 'NH4', 'Nit', 'NO2', 'PM2.5']
relation_dict = {'pos_correlate': 3.0,
                 'mixed_profile_sign': 2.0,
                 'neg_correlate': 1.0,
                 # 'na': 0
                 }
relation_list = [k for k in relation_dict]
# relation_list = ['NA'] + relation_list
# relation_list.append('NA')
relation_list.append('NA')

relation_inv_dict = {v: k for k, v in relation_dict.items()}


for idx, outcome in enumerate(outcome_list):
    file_list = find('*.csv', plot_dir+outcome+'/')
    for file in file_list:
        print(file)
        pollutant_profile = file.split('/')[-1].split('_')[2].split('  ')

        p_wo_thres = []
        relation_to_outcome = file.split('/')[-3]
        rel = relation_dict[relation_to_outcome]
        if len(pollutant_profile) == 1:
            set_p = pollutant_profile[0].split('>')[0].split('<')[0]
            if set_p in single_pollutantTimeWin:
                single_pollutantTimeWin[set_p][idx] = rel
            else:
                single_pollutantTimeWin[set_p] = 0.5*np.ones(len(outcome_list))
                single_pollutantTimeWin[set_p][idx] = rel
        else:
            for p in pollutant_profile:
                p_wo_thres.append(p.split('>')[0].split('<')[0])
            set_p = frozenset(p_wo_thres)

            if set_p in multi_pollutantTimeWin:
                multi_pollutantTimeWin[set_p][idx] = rel
            else:
                multi_pollutantTimeWin[set_p] = 0.5*np.ones(len(outcome_list))
                multi_pollutantTimeWin[set_p][idx] = rel



m_keys = multi_pollutantTimeWin.copy().keys()
if len(m_keys)>0:
    for key in m_keys:
        new_key = ''
        print(key)
        k_list = list(key)
        k_list.sort()
        print(k_list)
        for k in k_list:
            new_key += '{}_'.format(k)
        new_key = new_key[:-1]
        multi_pollutantTimeWin[new_key] = multi_pollutantTimeWin.pop(key)


single_df = pd.DataFrame.from_dict(single_pollutantTimeWin,
                                   orient='index',
                                   columns=outcome_list).sort_index()
multi_df = pd.DataFrame.from_dict(multi_pollutantTimeWin,
                                  orient='index',
                                  columns=outcome_list).sort_index()

print(single_df)
print(multi_df)
merged_df = pd.concat([single_df, multi_df])
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

fig, ax = plt.subplots(figsize=(6,18))
# ax = fig.add_subplot(111)
qrates = np.array(relation_list)
np_array = [0.5-1e-6, 0.5+1e-6, 1.5, 2.5, 3.5]
# norm = matplotlib.colors.BoundaryNorm(np.linspace(-0.5, 3.5, 5), 4)
norm = matplotlib.colors.BoundaryNorm(np_array, 4)
# norm = matplotlib.colors.BoundaryNorm(np.linspace(0.5, 3.5, 4), 3)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])
print(merged_df.values)
print(merged_df.columns)
print(merged_df.index)

im, _ = heatmap(merged_df.values, merged_df.index, [c.replace('_', '\n') for c in merged_df.columns], ax=ax,
                cmap=matplotlib.colors.ListedColormap(['white', 'green', 'blue', 'red']),
                norm=norm,
                cbar_kw=dict(ticks=np.arange(1, 4), format=fmt),
                cbarlabel="Relation between Profile and Outcome")

annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
                 textcolors=["red", "black"])
plt.show()
fig.savefig('heatmp_summary.png', bbox_inches="tight")

# output file
convert_str = {c: str for c in merged_df.columns}
merged_df = merged_df.astype(convert_str)

for k, v in relation_dict.items():
    merged_df = merged_df.replace(str(v), k)

print(merged_df.dtypes)
merged_df.to_csv('summary.csv')








