#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

def define_groups(dataframe, column): # define_groups_by_threshold
    """Define groups based on a threshold value in a specified column."""
    dataframe['group_split_2'] = dataframe[column].apply(lambda x: 1 if x <= 1 else 2 if x >= 9 else None)
    return dataframe

def get_bdi_group(score):
    """Determine BDI group based on score."""
    if score < 0 or score > 22:
        print('Score not found')
        return None
    return 1 if score < 4 else 2 if score < 9 else 3 if score < 13 else 4

def process_bdi(dataframe):
    """Process BDI responses from a dataframe."""
    relevant_columns = ['Experiment ID', 'Participant Public ID', 'Question Key', 'Response']
    response_keys = [f'response-{i}-quantised' for i in range(2, 9)]
    filtered_df = dataframe[dataframe['Question Key'].isin(response_keys)][relevant_columns]
    filtered_df['Response'] = filtered_df['Response'].astype(int) - 1
    
    control = filtered_df['Participant Public ID'].value_counts()
    print(control[control != 7])

    bdi_group_dict = {1: 'minimal', 2: 'mild', 3: 'moderate', 4: 'severe'}
    df_bdi = filtered_df.groupby('Participant Public ID')['Response'].sum().reset_index()
    df_bdi = df_bdi.rename(columns={'Response': 'bdi_score'})
    df_bdi['bdi_group_num'] = df_bdi['bdi_score'].apply(get_bdi_group)
    df_bdi['bdi_group'] = df_bdi['bdi_group_num'].map(bdi_group_dict)

    return df_bdi
