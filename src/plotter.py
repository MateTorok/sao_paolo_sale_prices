import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os



def save_fig(img_name, IMAGE_DIR = '.\\images' , fig_extension="png", res=300, verbose = False ):
    path = os.path.join(IMAGE_DIR, img_name + "." + fig_extension)
    if verbose:
        print("Saving figure", img_name)
    plt.savefig(path, format=fig_extension, dpi=res)


def corr_heatmap(df, figsize=(7.7,7.4)):
    corr_matrix = df.corr()
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
    sns.heatmap(corr_matrix, mask = mask, annot = True)
    plt.title('Correlation matrix heatmap')

def calc_square(full_lenght, cols):
    r, c = int(np.ceil(full_lenght / cols)), cols
    return r, c

def boxer(df, title='Boxplot', cols=3, figsize=(12,8)):
    r, c = calc_square(df.shape[1], cols=cols)
    fig, ax = plt.subplots(r, c, figsize=figsize)
    fig.suptitle(title)

    ax = ax.reshape(-1)
    for i, cols in enumerate(df.columns):
        sns.boxplot(df[cols], ax=ax[i] )
    for j in range(i+1, len(ax)):
        ax[j].set_axis_off() # Removing empty axes
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax



def make_histo(df, bins=35, title='Histogram', cols=3, figsize=(12,8) ):
    r, c = calc_square(df.shape[1], cols=cols)
    fig, ax = plt.subplots(r, c, figsize=figsize)

    fig.suptitle(title)
    ax = ax.reshape(-1)
    for i, cols in enumerate(df.columns):
        sns.distplot(df[cols], ax = ax[i])
    for j in range(i+1, len(ax)):
        ax[j].set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax

def act_vs_pred_plot(y_pred, y_train, figsize = (5,5) ):
    plt.figure(figsize=figsize)
    plt.plot(y_pred, y_train, 'o', alpha=0.6)
    plt.plot([0,1e7],[0,1e7], linestyle='-') # diagonal line
    plt.title('Actual and predicted price')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')