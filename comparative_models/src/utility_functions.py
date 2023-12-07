# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:32:41 2023

@author: carll
"""

# Utility functions 
def add_session_column(df, condition_col='condition'):
    
    """
    Adds a 'session' column to the DataFrame. The session number increments 
    each time the value in the condition_col changes from 'neut', 'pos', 
    or 'neg' to 'baseline'.

    :param df: Pandas DataFrame to which the session column will be added
    :param condition_col: Name of the column containing the condition values
    :return: DataFrame with the added 'session' column
    """
    
    session = 0
    session_numbers = []
    previous_condition = None  # Variable to store the previous condition

    for _, row in df.iterrows():
        current_condition = row[condition_col]
        if previous_condition in ['neut', 'pos', 'neg'] and current_condition == 'baseline':
            session += 1
        session_numbers.append(session)
        previous_condition = current_condition

    df['session'] = session_numbers
    
    return df