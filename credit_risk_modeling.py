#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue May 20 00:38:55 2025

@author: thonghuynh
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import polars as pl

# Class Credit Risk
class CreditRisk:
    def __init__(
        self, 
        portfolio : pd.core.frame.DataFrame, 
        loan_lifetime : int
    ):
        self.portfolio = portfolio
        self.loan_lifetime = loan_lifetime
        
    # Get IFRS stage depending on the days_past_due
    def get_IFRS9_stage(self, days_past_due):
        if days_past_due >= 90:
            stage = 3
        elif days_past_due >= 30:
            stage = 2
        else: 
            stage = 1   
        return stage
    def assign_stage(self, dpd_col='days_past_due'):
        self.portfolio['stage'] = self.portfolio[dpd_col].apply(
            lambda days_past_due : self.get_IFRS9_stage(days_past_due)    
        )
        return None
    def assign_CCF_from_stage(self, CCF_dict, stage_col='stage'):
        self.portfolio['CCF'] = self.portfolio['stage'].map(CCF_dict)
        return None
    
    def assign_PD_from_credit_rating(self, PD_dict, credit_rating_col='credit_rating'):
        self.portfolio['PD_12months'] = self.portfolio[credit_rating_col].map(PD_dict)
        return None
    
    def assign_LGD_from_collateral_type(self, LGD_dict, collateral_col='collateral_type'):
        self.portfolio['LGD'] = self.portfolio[collateral_col].map(LGD_dict)
        return None
    
    # Calculate EAD
    def calculate_EAD(self, portfolio, CCF_stressed_factor=1):
        portfolio['EAD'] = ( 
            portfolio['drawn_amount'] 
            + portfolio['undrawn_amount'] * portfolio['CCF'] * CCF_stressed_factor
        )
        return portfolio
    # Expected credit loss calculation
    def calculate_ECL(self, portfolio, LGD_stressed_factor=1, PD_stressed_factor=1):
        # Separate the stage
        stage_1 = portfolio.copy()[portfolio.stage == 1]
        stage_2 = portfolio.copy()[portfolio.stage == 2]
        stage_3 = portfolio.copy()[portfolio.stage == 3]
        
        # For stage 1, PD = PD_12months
        stage_1['ECL'] = stage_1['EAD'] * (
            stage_1['PD_12months'] * PD_stressed_factor
        ).clip(upper=1) * (
            stage_1['LGD'] * LGD_stressed_factor
        ).clip(upper=1)
        # For stage 2, needs to calculate the PD_lifetime
        stage_2['PD_lifetime'] = 1 - (1 - stage_2['PD_12months']) ** self.loan_lifetime
        stage_2['ECL'] = stage_2['EAD'] * (
            stage_2['PD_lifetime'] * PD_stressed_factor
        ).clip(upper=1) * (
            stage_2['LGD'] * LGD_stressed_factor
        ).clip(upper=1)
        stage_2 = stage_2.drop('PD_lifetime', axis=1)
        
        # For stage 3 of impairment, PD = 1
        PD_impairment = 1
        stage_3['ECL'] = stage_3['EAD'] * PD_impairment * (
            stage_3['LGD'] * LGD_stressed_factor
        ).clip(upper=1)
        
        # Concatenate all the stage
        portfolio = pd.concat([stage_1, stage_2, stage_3])
        # sort back by loan id
        portfolio = portfolio.set_index(['loan_id']).sort_index().reset_index()
        # assign
        return portfolio

if __name__ == '__main__':
    ## Random params
    np.random.seed(42)
    n_loans = 1000
    
    ## Credit risk params
    # Point-in-time PDs by rating
    PD_dict = {
        'AAA': 0.0001, 
        'AA': 0.0005, 
        'A': 0.001, 
        'BBB': 0.005, 
        'BB': 0.02, 
        'B': 0.05, 
        'CCC': 0.15
    }
    
    # LGD based on collateral
    LGD_dict = {'Secured': 0.35, 'Unsecured': 0.6}
    
    # CCF dict
    CCF_dict = {1: 0.5, 2: 0.8, 3: 0.8}
    
    # Stress scenarios
    # Stressed scenarios
    stressed_scenarios = {
        "normal" : {
            "CCF_stressed_factor" : 1, 
            "LGD_stressed_factor" : 1,
            "PD_stressed_factor" : 1
        },
        "distressed" : {
            "CCF_stressed_factor" : 1.2, 
            "LGD_stressed_factor" : 1,
            "PD_stressed_factor" : 1
        }  
        
    }
    
    # Simulate a synthetic portfolio
    portfolio = pd.DataFrame({
        'loan_id': range(1, n_loans + 1),
        'exposure': np.random.uniform(100_000, 5_000_000, n_loans),
        'credit_rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'], n_loans),
        'collateral_type': np.random.choice(['Secured', 'Unsecured'], n_loans, p=[0.4, 0.6]),
        'days_past_due': np.random.choice([0, 15, 30, 60, 90], n_loans, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    })
    # Loan lifetime of this portfolio
    loan_lifetime = 5
    
    # Add drawn and undrawn amounts
    portfolio['drawn_amount'] = portfolio['exposure'] * np.random.uniform(0.5, 1.0, n_loans)
    portfolio['undrawn_amount'] = portfolio['exposure'] * np.random.uniform(0.0, 0.5, n_loans)
    
    # CCF assumptions
    #ccf_baseline = 0.6
    #ccf_stressed = 0.8
    
    # Instantiate object
    credit_risk = CreditRisk(portfolio, loan_lifetime)
    
    # Stage assignment
    credit_risk.assign_stage(dpd_col='days_past_due')
    
    # Assign CCF
    credit_risk.assign_CCF_from_stage(CCF_dict)
    
    # Assign PD
    credit_risk.assign_PD_from_credit_rating(PD_dict)
    
    # Assign LGD
    credit_risk.assign_LGD_from_collateral_type(LGD_dict)

    # Calculate EAD
    credit_risk.portfolio = credit_risk.calculate_EAD(
        credit_risk.portfolio,
        CCF_stressed_factor=1
    )    
    
    # Calculate ECL
    credit_risk.portfolio = credit_risk.calculate_ECL(
        credit_risk.portfolio,
        LGD_stressed_factor=1.2,
        PD_stressed_factor=1.5
    )    
    