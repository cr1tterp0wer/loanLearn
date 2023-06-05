import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#########################
# Data Visualization
########################

# Correlation Matrix
def corr_matrix(df):
    cor_fig, cor_ax = plt.subplots(figsize=(15, 10))
    corr_matrix = df.corr(numeric_only=True)
    cor_ax = sns.heatmap(corr_matrix, annot=True)
    plt.xticks(rotation=30, horizontalalignment='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()

# Total/Average disbursed loan amt by industry
def avg_amt_by_industry(df, plt):
    industry_group = df.groupby(['Industry'])
    df_industrySum = industry_group.sum(numeric_only=True).sort_values('DisbursementGross', ascending=False)
    df_industryAve = industry_group.mean(numeric_only=True).sort_values('DisbursementGross', ascending=False)
    fig = plt.figure(figsize=(25, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # Gross SBA Loan Disbursement by Industry
    ax1.bar(df_industrySum.index, df_industrySum['DisbursementGross'] / 1000000000)
    ax1.set_xticklabels(df_industrySum.index, rotation=30, horizontalalignment='right', fontsize=10)

    ax1.set_title('Gross SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
    ax1.set_xlabel('Industry')
    ax1.set_ylabel('Gross Loan Disbursement (Billions)')

    # Average SBA Loan Disbursement by Industry
    ax2.bar(df_industryAve.index, df_industryAve['DisbursementGross'])
    ax2.set_xticklabels(df_industryAve.index, rotation=30, horizontalalignment='right', fontsize=10)

    ax2.set_title('Average SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
    ax2.set_xlabel('Industry')
    ax2.set_ylabel('Average Disbursement (Billions)')
    plt.show()


# Average days to disbursement by Industry
def avg_dispersement_delay(df, plt):
    fig2, ax = plt.subplots(figsize=(15,5))

    ax.bar(df_industryAve.index, df_industryAve['DaysToDisbursement'].sort_values(ascending=False))
    ax.set_xticklabels(df_industryAve['DaysToDisbursement'].sort_values(ascending=False).index, rotation=35, horizontalalignment='right', fontsize=10)

    ax.set_title('Average Days to SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
    ax.set_xlabel('Industry')
    ax.set_ylabel('Average Days to Disbursement')
    plt.show()

# Function for creating stacked bar charts grouped by desired column
# df = original data frame, col = x-axis grouping, stack_col = column to show stacked values
# Essentially acts as a stacked histogram when stack_col is a flag variable
def stacked_setup(df, col, axes, stack_col='Default'):
    data = df.groupby([col, stack_col])[col].count().unstack(stack_col)
    data.fillna(0)
    axes.bar(data.index, data[1], label='Default')
    axes.bar(data.index, data[0], bottom=data[1], label='Paid in full')

# Number of PIF/Defaulted Loans by Industry from 1984-2010
def total_loan_resolutions(df, plt):
    fig3 = plt.figure(figsize=(15, 10))
    ax1a = plt.subplot(2,1,1)
    ax2a = plt.subplot(2,1,2)

    # Number of Paid in full and defaulted loans by industry
    stacked_setup(df, col='Industry', axes=ax1a)
    ax1a.set_xticklabels(df.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default').index,
                         rotation=35, horizontalalignment='right', fontsize=10)

    ax1a.set_title('Number of PIF/Defaulted Loans by Industry from 1984-2010', fontsize=15)
    ax1a.set_xlabel('Industry')
    ax1a.set_ylabel('Number of PIF/Defaulted Loans')
    ax1a.legend()

    # Number of Paid in full and defaulted loans by State
    stacked_setup(df, col='State', axes=ax2a)

    ax2a.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
    ax2a.set_xlabel('State')
    ax2a.set_ylabel('Number of PIF/Defaulted Loans')
    ax2a.legend()

    plt.tight_layout()
    plt.show()

# Paid in full and Defaulted loans by DisbursementFY
def pif_vs_defaulted_by_disbursementFY(df, plt):
    fig4, ax4 = plt.subplots(figsize=(15, 5))
    stack_data = df.groupby(['DisbursementFY', 'Default'])['DisbursementFY'].count().unstack('Default')
    x = stack_data.index
    y = [stack_data[1], stack_data[0]]
    ax4.stackplot(x, y, labels=['Default', 'Paid in full'])
    ax4.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
    ax4.set_xlabel('Disbursement Year')
    ax4.set_ylabel('Number of PIF/Defaulted Loans')
    ax4.legend(loc='upper left')

# Paid in full and defaulted loans backed by Real Estate
def pif_vs_defaulted_by_real_estate(df, plt):
    fig5 = plt.figure(figsize=(20, 10))
    ax1b = fig5.add_subplot(1, 2, 1)
    ax2b = fig5.add_subplot(1, 2, 2)

    stacked_setup(df=df, col='RealEstate', axes=ax1b)
    ax1b.set_xticks(df.groupby(['RealEstate', 'Default'])['RealEstate'].count().unstack('Default').index)
    ax1b.set_xticklabels(labels=['No', 'Yes'])

    ax1b.set_title('Number of PIF/Defaulted Loans backed by Real Estate from 1984-2010', fontsize=15)
    ax1b.set_xlabel('Loan Backed by Real Estate')
    ax1b.set_ylabel('Number of Loans')
    ax1b.legend()

    # Paid in full and defaulted loans active during the Great Recession
def pif_vs_defaulted_recession(df, plt):
    stacked_setup(df=df, col='GreatRecession', axes=ax2b)
    ax2b.set_xticks(df.groupby(['GreatRecession', 'Default'])['GreatRecession'].count().unstack('Default').index)
    ax2b.set_xticklabels(labels=['No', 'Yes'])
    ax2b.set_title('Number of PIF/Defaulted Loans Active during the Great Recession from 1984-2010', fontsize=15)
    ax2b.set_xlabel('Loan Active during Great Recession')
    ax2b.set_ylabel('Number of Loans')
    ax2b.legend()

# Default percentage by Industry
def industry_ratio(df):
    def_ind = df.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default')
    def_ind['Def_Percent'] = def_ind[1]/(def_ind[1] + def_ind[0])
    print(def_ind)

# Default percentage by State
def state_ratio(df):
    def_state = df.groupby(['State', 'Default'])['State'].count().unstack('Default')
    def_state['Def_Percent'] = def_state[1]/(def_state[1] + def_state[0])
    print(def_state)

# Check Default percentage for loans backed by Real Estate
def default_ratio_real_estate(df):
    def_re = df.groupby(['RealEstate', 'Default'])['RealEstate'].count().unstack('Default')
    def_re['Def_Percent'] = def_re[1]/(def_re[1] + def_re[0])
    print(def_re)

# Check Default percentage for loans active during the Great Recession
def default_ratio_recession(df):
    def_gr = df.groupby(['GreatRecession', 'Default'])['GreatRecession'].count().unstack('Default')
    def_gr['Def_Percent'] = def_gr[1]/(def_gr[1] + def_gr[0])
    print(def_gr)

