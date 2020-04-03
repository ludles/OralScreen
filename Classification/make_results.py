# -*- coding: utf-8 -*-
# make tables for the paper from csv files in 'results' folder

from glob import glob
import os, csv
import numpy as np
import pandas as pd

# %%
def make_plot_table(csv_path, res_root):
    dir_name = os.path.dirname(csv_path)
    tar_path = res_root + f"plots/plot_{os.path.split(dir_name)[-1]}_" + os.path.basename(csv_path)
    if not os.path.exists(os.path.dirname(tar_path)):
        os.makedirs(os.path.dirname(tar_path))
    
    df = pd.read_csv(csv_path).drop(columns='slide_accuracy')
    df_grouped = df.groupby(['fold','i_slide']).mean()
    df_grouped['std'] = df.groupby(['fold','i_slide']).std()['perc_cancer']
#    df_grouped['fold'] = df_grouped.index.get_level_values(level='fold')
#    df_grouped = df_grouped.droplevel('fold')
#    df_grouped.loc[2].loc[12]['perc_cancer']
    
#    df_sc = df.loc[:, ['i_slide', 'slide_class']]
    
#    df_cancer = df[df['slide_class'] == 'Cancer'].groupby(['fold','i_slide']).mean()
    l_cancer = list(set(df[df['slide_class'] == 'Cancer']['i_slide']))
    l_cancer.sort()
    
#    df_healthy = df[df['slide_class'] == 'Healthy'].groupby(['fold','i_slide']).mean()
    l_healthy = list(set(df[df['slide_class'] == 'Healthy']['i_slide']))
    l_healthy.sort()
    
    slides = l_healthy + l_cancer
    folds = list(set(df['fold']))
    folds.sort()
    
    aves = [[0] * len(slides) for i in range(len(folds))]
    stds = [[0] * len(slides) for i in range(len(folds))]
    
    for fold in folds:
        for slide in slides:
            try:
                aves[fold - 1][slides.index(slide)] = df_grouped.loc[fold].loc[slide]['perc_cancer']
            except:
                pass
            try:
                stds[fold - 1][slides.index(slide)] = df_grouped.loc[fold].loc[slide]['std']
            except:
                pass
    
    with open(tar_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + slides + ['STDs'])
        for fold in folds:
            writer.writerow([f'fold{fold}'] + aves[fold - 1] + stds[fold - 1])
    
    return

# %%

res_root='./results/'

csv_paths = [os.path.join(root, file) for (root, dirs, files) in os.walk(res_root+'per_slide') for file in files]

for csv_path in csv_paths:
    make_plot_table(csv_path, res_root)


# %%

def make_results_table(res_root, tab_name):
    '''
    make table in the paper from csv files in 'results' folder
    '''
    for dataset in ['1', '2', '3']:
        dir_res_dataset = res_root + f'dataset_{dataset}/'
        # results of each dataset
        csv_paths = glob(dir_res_dataset + '*.csv')
        csv_paths.sort(key = lambda i:len(i))
        
        for csv_path in csv_paths:
            # results of each method
            method_name = os.path.basename(csv_path)[:-len('.csv')]
            network_name = method_name.split('_')[0]
            if 'pre1' in method_name:
                network_name += '(pre-trained)'
    #        print(network_name)
    
            # calculate metrics                
            df = pd.read_csv(csv_path).loc[:, ['fold', 'i_model','TP', 'FP', 'TN', 'FN']]
            df = df.groupby('i_model').sum()
            df['tot'] = df.loc[:,['TP', 'FP', 'TN', 'FN']].sum(axis=1)
            df['acc'] = (df['TP'] + df['TN']) / df['tot']
            df['pr'] = df['TP'] / (df['TP'] + df['FP'])
            df['re'] = df['TP'] / (df['TP'] + df['FN'])
            df['f1'] = 2 * df['pr'] * df['re'] / (df['pr'] + df['re'])
            df_res = pd.DataFrame({'mean': df.loc[:,['acc', 'pr', 're', 'f1']].mean(), 
                                   'std': df.loc[:,['acc', 'pr', 're', 'f1']].std()
                                   })
            
            # output formatting
            accuracy = f"{df_res['mean']['acc']*100:.1f}$\pm${df_res['std']['acc']*100:.1f}"
            precision = f"{df_res['mean']['pr']*100:.1f}$\pm${df_res['std']['pr']*100:.1f}"
            recall = f"{df_res['mean']['re']*100:.1f}$\pm${df_res['std']['re']*100:.1f}"
            f1 = f"{df_res['mean']['f1']*100:.1f}$\pm${df_res['std']['f1']*100:.1f}"
            
            tab_header = ['Dataset', 'Network', 'Accuracy', 'Precision', 'Recall', 'F1-score']
            if not os.path.exists(res_root + tab_name):
                with open(res_root + tab_name, 'a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(tab_header)
            with open(res_root + tab_name, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dataset, network_name, accuracy, precision, recall, f1])
    return
# %%
make_results_table(res_root='./results/', tab_name='results_table_all.csv')
