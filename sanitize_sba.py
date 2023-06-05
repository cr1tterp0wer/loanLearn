# UNUSED ATTRIBUTES
#   LoanNr_ChkDgt and Name - provides no value to the actual analysis
#   City and Zip - each have a large number of unique values, and my assumption is that it is not likely either would have any particularly significant values
#   Bank - Name of the bank shouldn't matter for analysis, however this could potentially be used when revisiting this analysis to determine the asset size of the bank servicing the loan
#   ChgOffDate - only applies when a loan is charged off and isn't relevant to the analysis
#   NAICS - replaced by Industry
#   NewExist - replaced by NewBusiness flag field
#   FranchiseCode - replaced by IsFranchise flag field
#   ApprovalDate and DisbursementDate - hypothesis that DaysToDisbursement will be more valueable
#   SBA_Appv - guaranteed amount is based on percentage of gross loan amount, not dollar amount typically
#   MIS_Status - Default field replaces this as the target field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ApprovalFY has more than one type int|object
def sanitize_approval(x):
    if isinstance(x, str):
        return x.replace('A', '')
    return x

def cleanup(df):
    df.copy

    # Remove Null values from columns
    df.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist', 'RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)

    # Replace currency symbols
    df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',',''))
    df['ApprovalFY'] = df['ApprovalFY'].apply(sanitize_approval).astype('int64')
    df = df.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float', 'BalanceGross': 'float', 'ChgOffPrinGr': 'float', 'GrAppv': 'float', 'SBA_Appv': 'float'})

    # NAIC Codes describe business
    # First 2 digits specify industry, the rest scope in on subtypes
    # Only interested in first 2 digits for this model
    df['Industry'] = df['NAICS'].astype('str').apply(lambda x: x[:2])

    # Map the industry to the first two digits
    df['Industry'] = df['Industry'].map({
        '11': 'Ag/For/Fish/Hunt',
        '21': 'Min/Quar/Oil_Gas_ext',
        '22': 'Utilities',
        '23': 'Construction',
        '31': 'Manufacturing',
        '32': 'Manufacturing',
        '33': 'Manufacturing',
        '42': 'Wholesale_trade',
        '44': 'Retail_trade',
        '45': 'Retail_trade',
        '48': 'Trans/Ware',
        '49': 'Trans/Ware',
        '51': 'Information',
        '52': 'Finance/Insurance',
        '53': 'RE/Rental/Lease',
        '54': 'Prof/Science/Tech',
        '55': 'Mgmt_comp',
        '56': 'Admin_sup/Waste_Mgmt_Rem',
        '61': 'Educational',
        '62': 'Healthcare/Social_assist',
        '71': 'Arts/Entertain/Rec',
        '72': 'Accom/Food_serv',
        '81': 'Other_no_pub',
        '92': 'Public_Admin'
    })

    # Remove records whose NAICS is not supported (0)
    df.dropna(subset=['Industry'], inplace=True)

    # Simpy classify if a business is a Franchise or not
    df.loc[(df['FranchiseCode'] <= 1), 'IsFranchise'] = 0
    df.loc[(df['FranchiseCode'] > 1), 'IsFranchise'] = 1

    # Ensure NewExist only has 1s or 2s; remove record otherwise
    df = df[(df['NewExist'] == 1) | (df['NewExist'] == 2)]
    df.loc[(df['NewExist'] == 1), 'NewBusiness'] = 0
    df.loc[(df['NewExist'] == 2), 'NewBusiness'] = 1

    # Remove records where RevLineCr != Y|N
    df = df[(df['RevLineCr'] == 'Y') | (df['RevLineCr'] == 'N')]

    # Remove records where LowDoc != Y|N
    df = df[(df['LowDoc'] == 'Y') | (df['LowDoc'] == 'N')]

    # RevLineCr & LowDoc: 0 = N, 1 = Y
    df['RevLineCr'] = np.where(df['RevLineCr'] == 'N', 0, 1)
    df['LowDoc'] = np.where(df['LowDoc'] == 'N', 0, 1)

    # MIS_Status
    # PIF: Paid In Full
    # CHGOFF: default
    df['Default'] = np.where(df['MIS_Status'] == 'P I F', 0, 1)

    # Convert ApprovalDate & DispersementDate columns to datetime values
    # ChgOffDate not changed to datetime since it is not valuable: DEPRECATED
    df[['ApprovalDate', 'DisbursementDate']] = \
        df[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)

    # Enhance Records
    # Add new field to track time it took to disburse loan to business
    df['DaysToDisbursement'] = df['DisbursementDate'] - df['ApprovalDate']

    # Convert DaysToDisbursement from a timedelta64 dtype to an int64
    # Strip 'days' and convert to int
    df['DaysToDisbursement'] = df['DaysToDisbursement'] \
        .astype('str') \
        .apply(lambda x: x[:x.index('d')-1]).astype('int64')

    # Create DisbursementFY field for time selection later
    df['DisbursementFY'] = df['DisbursementDate'].map(lambda x: x.year)


    # It may be hard to service a loan out of state
    # Generate a flag if business is out of state
    df['StateSame'] = np.where(df['State'] == df['BankState'], 1, 0)


    # It is normal for SBA to guarantee a % of the loan in the even of a loss
    # if the bussiness is unable to pay. This increases the risk to SBA
    # Create new col to capture this data
    df['SBA_AppvPct'] = df['SBA_Appv'] / df['GrAppv']

    # Sometimes the full ammount of the loan is not approved,
    # Capture this data
    df['AppvDisbursed'] = np.where(df['DisbursementGross'] == df['GrAppv'], 1, 0)

    df = df.astype({'IsFranchise': 'int64', 'NewBusiness': 'int64'})

# Remove deprecated attributes
def remove_unused(df):
    df.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
        'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv', 'MIS_Status'], inplace=True)


# Great Recession (2007-2009) Impact
def enhance_greate_depression(df):
    df['RealEstate'] = np.where(df['Term'] >= 240, 1, 0)

    # Generate col for Great Recession
    df['GreatRecession'] = np.where(((2007 <= df['DisbursementFY']) & (df['DisbursementFY'] <= 2009)) |
                                     ((df['DisbursementFY'] < 2007) & (df['DisbursementFY'] + (df['Term']/12) >= 2007)), 1, 0)

    # Analyze loans disbursed through 2010, account for Great Recession, restrict the year
    # Select only records with a disbursement year through 2010
    df = df[df['DisbursementFY'] <= 2010]

    # Create flag to signify if a larger amount was disbursed than what the Bank had approved
    # Likely RevLineCr?
    df['DisbursedGreaterAppv'] = np.where(df['DisbursementGross'] > df['GrAppv'], 1, 0)

    # Remove records with loans disbursed prior to being approved
    df = df[df['DaysToDisbursement'] >= 0]
