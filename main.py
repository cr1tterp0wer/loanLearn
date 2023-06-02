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

# Import and analyze data
df = pd.read_csv('./data/SBAnational.csv')
df_copy = df.copy()

# print(df_copy.shape)
# print(df_copy.isnull().sum())

# Remove Null values from columns
df_copy.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist', 'RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)

# Replace currency symbols
df_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = df_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',',''))

# ApprovalFY has more than one type int|object
def sanitize_approval(x):
    if isinstance(x, str):
        return x.replace('A', '')
    return x

df_copy['ApprovalFY'] = df_copy['ApprovalFY'].apply(sanitize_approval).astype('int64')
df_copy = df_copy.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float', 'BalanceGross': 'float', 'ChgOffPrinGr': 'float', 'GrAppv': 'float', 'SBA_Appv': 'float'})

# NAIC Codes describe business
# First 2 digits specify industry, the rest scope in on subtypes
# Only interested in first 2 digits for this model
df_copy['Industry'] = df_copy['NAICS'].astype('str').apply(lambda x: x[:2])

# Map the industry to the first two digits
df_copy['Industry'] = df_copy['Industry'].map({
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
df_copy.dropna(subset=['Industry'], inplace=True)

# Simpy classify if a business is a Franchise or not
df_copy.loc[(df_copy['FranchiseCode'] <= 1), 'IsFranchise'] = 0
df_copy.loc[(df_copy['FranchiseCode'] > 1), 'IsFranchise'] = 1

# Ensure NewExist only has 1s or 2s; remove record otherwise
df_copy = df_copy[(df_copy['NewExist'] == 1) | (df_copy['NewExist'] == 2)]
df_copy.loc[(df_copy['NewExist'] == 1), 'NewBusiness'] = 0
df_copy.loc[(df_copy['NewExist'] == 2), 'NewBusiness'] = 1


# Double check RevLineCr and LowDoc
# print(df_copy['RevLineCr'].unique())
# print(df_copy['LowDoc'].unique())

# Remove records where RevLineCr != Y|N
df_copy = df_copy[(df_copy['RevLineCr'] == 'Y') | (df_copy['RevLineCr'] == 'N')]

# Remove records where LowDoc != Y|N
df_copy = df_copy[(df_copy['LowDoc'] == 'Y') | (df_copy['LowDoc'] == 'N')]

# RevLineCr & LowDoc: 0 = N, 1 = Y
df_copy['RevLineCr'] = np.where(df_copy['RevLineCr'] == 'N', 0, 1)
df_copy['LowDoc'] = np.where(df_copy['LowDoc'] == 'N', 0, 1)

# MIS_Status
# PIF: Paid In Full
# CHGOFF: default
df_copy['Default'] = np.where(df_copy['MIS_Status'] == 'P I F', 0, 1)

# Convert ApprovalDate & DispersementDate columns to datetime values
# ChgOffDate not changed to datetime since it is not valuable: DEPRECATED
df_copy[['ApprovalDate', 'DisbursementDate']] = df_copy[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)

# Enhance Records
# Add new field to track time it took to disburse loan to business
df_copy['DaysToDisbursement'] = df_copy['DisbursementDate'] - df_copy['ApprovalDate']

# Convert DaysToDisbursement from a timedelta64 dtype to an int64
# Strip 'days' and convert to int
df_copy['DaysToDisbursement'] = df_copy['DaysToDisbursement'].astype('str').apply(lambda x: x[:x.index('d')-1]).astype('int64')

# Create DisbursementFY field for time selection later
df_copy['DisbursementFY'] = df_copy['DisbursementDate'].map(lambda x: x.year)


# It may be hard to service a loan out of state
# Generate a flag if business is out of state
df_copy['StateSame'] = np.where(df_copy['State'] == df_copy['BankState'], 1, 0)


# It is normal for SBA to guarantee a % of the loan in the even of a loss
# if the bussiness is unable to pay. This increases the risk to SBA
# Create new col to capture this data
df_copy['SBA_AppvPct'] = df_copy['SBA_Appv'] / df_copy['GrAppv']

# Sometimes the full ammount of the loan is not approved,
# Capture this data
df_copy['AppvDisbursed'] = np.where(df_copy['DisbursementGross'] == df_copy['GrAppv'], 1, 0)

df_copy = df_copy.astype({'IsFranchise': 'int64', 'NewBusiness': 'int64'})

# Remove deprecated attributes
# LoanNr_ChkDgt and Name - provides no value to the actual analysis
# City and Zip - each have a large number of unique values, and my assumption is that it is not likely either would have any particularly significant values
# Bank - Name of the bank shouldn't matter for analysis, however this could potentially be used when revisiting this analysis to determine the asset size of the bank servicing the loan
# ChgOffDate - only applies when a loan is charged off and isn't relevant to the analysis
# NAICS - replaced by Industry
# NewExist - replaced by NewBusiness flag field
# FranchiseCode - replaced by IsFranchise flag field
# ApprovalDate and DisbursementDate - hypothesis that DaysToDisbursement will be more valueable
# SBA_Appv - guaranteed amount is based on percentage of gross loan amount, not dollar amount typically
# MIS_Status - Default field replaces this as the target field

df_copy.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
    'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv', 'MIS_Status'], inplace=True)

# Find RealEstate backed loans, 
# Find out if they were impacted by the Great Recession (2007-2009)
df_copy['RealEstate'] = np.where(df_copy['Term'] >= 240, 1, 0)

# Generate col for Great Recession
df_copy['GreatRecession'] = np.where(((2007 <= df_copy['DisbursementFY']) & (df_copy['DisbursementFY'] <= 2009)) |
                                     ((df_copy['DisbursementFY'] < 2007) & (df_copy['DisbursementFY'] + (df_copy['Term']/12) >= 2007)), 1, 0)

# Analyze loans disbursed through 2010, account for Great Recession, restrict the year
# Select only records with a disbursement year through 2010
df_copy = df_copy[df_copy['DisbursementFY'] <= 2010]

# Create flag to signify if a larger amount was disbursed than what the Bank had approved
# Likely RevLineCr?
df_copy['DisbursedGreaterAppv'] = np.where(df_copy['DisbursementGross'] > df_copy['GrAppv'], 1, 0)

# Remove records with loans disbursed prior to being approved
df_copy = df_copy[df_copy['DaysToDisbursement'] >= 0]

#########################
# Data Visualization
########################

# Correlation Matrix
cor_fig, cor_ax = plt.subplots(figsize=(15, 10))
corr_matrix = df_copy.corr(numeric_only=True)
cor_ax = sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=30, horizontalalignment='right', fontsize=8)
plt.yticks(fontsize=8)

# Total/Average disbursed loan amt by industry
industry_group = df_copy.groupby(['Industry'])

# Data frames
df_industrySum = industry_group.sum(numeric_only=True).sort_values('DisbursementGross', ascending=False)
df_industryAve = industry_group.mean(numeric_only=True).sort_values('DisbursementGross', ascending=False)

# Figure to place charts in GUI
fig = plt.figure(figsize=(25, 5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

#
# Bar chart 1
# Gross SBA Loan Disbursement by Industry
#
ax1.bar(df_industrySum.index, df_industrySum['DisbursementGross'] / 1000000000)
ax1.set_xticklabels(df_industrySum.index, rotation=30, horizontalalignment='right', fontsize=10)

ax1.set_title('Gross SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax1.set_xlabel('Industry')
ax1.set_ylabel('Gross Loan Disbursement (Billions)')

#
# Bar chart 2
# Average SBA Loan Disbursement by Industry
#
ax2.bar(df_industryAve.index, df_industryAve['DisbursementGross'])
ax2.set_xticklabels(df_industryAve.index, rotation=30, horizontalalignment='right', fontsize=10)

ax2.set_title('Average SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax2.set_xlabel('Industry')
ax2.set_ylabel('Average Disbursement (Billions)')


#
# Bar chart 3
# Average days to disbursement by Industry
#

fig2, ax = plt.subplots(figsize=(15,5))

ax.bar(df_industryAve.index, df_industryAve['DaysToDisbursement'].sort_values(ascending=False))
ax.set_xticklabels(df_industryAve['DaysToDisbursement'].sort_values(ascending=False).index, rotation=35, horizontalalignment='right', fontsize=10)

ax.set_title('Average Days to SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax.set_xlabel('Industry')
ax.set_ylabel('Average Days to Disbursement')


#
# Chart 4
# Stacked Bar
# Number of PIF/Defaulted Loans by Industry from 1984-2010
#
fig3 = plt.figure(figsize=(15, 10))

ax1a = plt.subplot(2,1,1)
ax2a = plt.subplot(2,1,2)

# Function for creating stacked bar charts grouped by desired column
# df = original data frame, col = x-axis grouping, stack_col = column to show stacked values
# Essentially acts as a stacked histogram when stack_col is a flag variable
def stacked_setup(df, col, axes, stack_col='Default'):
    data = df.groupby([col, stack_col])[col].count().unstack(stack_col)
    data.fillna(0)
    axes.bar(data.index, data[1], label='Default')
    axes.bar(data.index, data[0], bottom=data[1], label='Paid in full')

# Number of Paid in full and defaulted loans by industry
stacked_setup(df=df_copy, col='Industry', axes=ax1a)
ax1a.set_xticklabels(df_copy.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default').index,
                     rotation=35, horizontalalignment='right', fontsize=10)

ax1a.set_title('Number of PIF/Defaulted Loans by Industry from 1984-2010', fontsize=15)
ax1a.set_xlabel('Industry')
ax1a.set_ylabel('Number of PIF/Defaulted Loans')
ax1a.legend()

# Number of Paid in full and defaulted loans by State
stacked_setup(df=df_copy, col='State', axes=ax2a)

ax2a.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
ax2a.set_xlabel('State')
ax2a.set_ylabel('Number of PIF/Defaulted Loans')
ax2a.legend()

plt.tight_layout()


# Default percentage by Industry
def_ind = df_copy.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default')
def_ind['Def_Percent'] = def_ind[1]/(def_ind[1] + def_ind[0])
# print(def_ind)

# Default percentage by State
def_state = df_copy.groupby(['State', 'Default'])['State'].count().unstack('Default')
def_state['Def_Percent'] = def_state[1]/(def_state[1] + def_state[0])
# print(def_state)



# Paid in full and Defaulted loans by DisbursementFY
# Decided to use a stacked area chart here since it's time series data
fig4, ax4 = plt.subplots(figsize=(15, 5))

stack_data = df_copy.groupby(['DisbursementFY', 'Default'])['DisbursementFY'].count().unstack('Default')
x = stack_data.index
y = [stack_data[1], stack_data[0]]

ax4.stackplot(x, y, labels=['Default', 'Paid in full'])
ax4.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
ax4.set_xlabel('Disbursement Year')
ax4.set_ylabel('Number of PIF/Defaulted Loans')
ax4.legend(loc='upper left')

# Paid in full and defaulted loans backed by Real Estate
fig5 = plt.figure(figsize=(20, 10))

ax1b = fig5.add_subplot(1, 2, 1)
ax2b = fig5.add_subplot(1, 2, 2)

stacked_setup(df=df_copy, col='RealEstate', axes=ax1b)
ax1b.set_xticks(df_copy.groupby(['RealEstate', 'Default'])['RealEstate'].count().unstack('Default').index)
ax1b.set_xticklabels(labels=['No', 'Yes'])

ax1b.set_title('Number of PIF/Defaulted Loans backed by Real Estate from 1984-2010', fontsize=15)
ax1b.set_xlabel('Loan Backed by Real Estate')
ax1b.set_ylabel('Number of Loans')
ax1b.legend()

# Paid in full and defaulted loans active during the Great Recession
stacked_setup(df=df_copy, col='GreatRecession', axes=ax2b)
ax2b.set_xticks(df_copy.groupby(['GreatRecession', 'Default'])['GreatRecession'].count().unstack('Default').index)
ax2b.set_xticklabels(labels=['No', 'Yes'])

ax2b.set_title('Number of PIF/Defaulted Loans Active during the Great Recession from 1984-2010', fontsize=15)
ax2b.set_xlabel('Loan Active during Great Recession')
ax2b.set_ylabel('Number of Loans')
ax2b.legend()

# Check Default percentage for loans backed by Real Estate
def_re = df_copy.groupby(['RealEstate', 'Default'])['RealEstate'].count().unstack('Default')
def_re['Def_Percent'] = def_re[1]/(def_re[1] + def_re[0])
# print(def_re)

# Check Default percentage for loans active during the Great Recession
def_gr = df_copy.groupby(['GreatRecession', 'Default'])['GreatRecession'].count().unstack('Default')
def_gr['Def_Percent'] = def_gr[1]/(def_gr[1] + def_gr[0])
# print(def_gr)

##
# Render Data Visualization
##
# plt.show()

#########################
# AI Modeling
########################

# LOGISTICAL REGRESSION MODEL

# Use ONE-HOT encoding for data attributes
df_copy = pd.get_dummies(df_copy)
#print(df_copy.head())

# Establish target and feature fields
# X: input into the model
# Y: Output from the model
# Make predictions on X_Val
# Test the predictions against Y_val

# y = df_copy['Default']
# X = df_copy.drop('Default', axis=1)

# Scale the feature values prior to modeling
# Overlapping two standard deviations on top of each other
# StandardScaler():z = (x - u) / s
# @params u: mean of training samples
# @params s: standard deviation of traning samples
# scale = StandardScaler()
# X_scaled = scale.fit_transform(X)

# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25)

# Init model
# log_reg = LogisticRegression(random_state=2)

# Train the model and make predictions
# log_reg.fit(X_train, y_train)
# y_logpred = log_reg.predict(X_val)

# Print the results
# print("Logistical Regression Model")
# print(classification_report(y_val, y_logpred, digits=3))

# XGBoost REGRESSION MODEL

y = df_copy['Default']
X = df_copy.drop('Default', axis=1)

scale = StandardScaler()
X_scaled = scale.fit_transform(X)
print(X_scaled)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25)

xgboost = XGBClassifier(random_state=2)

# Create our Trees
xgboost.fit(X_train, y_train)
y_xgbpred = xgboost.predict(X_val)

# Print the results
print("XGB Classifier Model")
print(classification_report(y_val, y_xgbpred, digits=3))

# Save our model and use it for other SBA apps
xgboost.save_model("./sba_default.xgb")

# List the importance of each feature
# print("Feature Importance")
# for name, importance in sorted(zip(X.columns, xgboost.feature_importances_)):
#     print(name, "=", importance)

# OUTPUT
# TP: True Positive
# TN: True Negative
# FN: False Negative
# FP: False Positive
# @params precision: positive predictive value, TP/(TP+FP)
# @params recall: (TPR) True Positive Rate, percentage of data correctly identified; TP/(TP+FN)
# @params f1-score: Accuracy
# @params support: Frequency a rule appears in the data
# @params accuracy: Percentage of correct predictions, (TP+TN)/(TP+TN+FP+FN)
# @params 0: class_0: Default = 0
# @params 1: class_1: Default = 1
