#%%
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


#%%[markdown]
# This project is about Vehicle Insurance fraud detection. From the dataset the main factor that we have choosen in fraud, which speaks about two cases: 1) Whether the insurance claimed is fraud, 2) Whether the insurance claimed is not fraud. As a team when we discussed, without even diving into analysis and by just looking at the data provided we thought of a few factors that may define if a case is fraud or not. They are:
#Policy Type - Actual policy holder or Third Party
# * Vehicle category and price
# * History of claims
# * Age of Vehicle
# * Time between the incident and claimed
# * Location of accident
# * Accident details
# * police report
# * driver rating
# * Number of cars involved
# * Year of policy
##
# With this in our minds we went ahead with our project to see if these actually the factors or not or are there any other values in our dataset that will get added.
# %% [markdown]
#loading the dataset
# Printing the total values in the dataset and the few rows
file_path = r'/Users/bhumikamallikarjunhorapet/Documents/GitHub/DATS-6103-FA-23-SEC-11-TEAM2/Datasets/fraud_oracle.csv' ##enter your path here

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
print('The data constain ', len(df),' observations.')
df.head()


# %%[markdown]
# Finding all the null values
print(df.isnull().sum())


# %%[markdown]
# Printing all the different datatypes present.
df.describe()
df.dtypes
# %%[markdown]
# Printing all the unique values present.
for column in df.columns:
    unique_values = df[column].unique()
    print(f"'{column}': {unique_values}\n")

# %%[markdown]
# We can see that Month and Day of week claimed have 0 in them.
#1. Investigating that!
print('DayOfWeekClaimed has ', len(df.loc[(df['DayOfWeekClaimed']=='0')]), ' row(s) with a 0')
print('MonthClaimed has ',len(df.loc[(df['MonthClaimed']=='0')]),' row(s) with a 0') 
print(' ')

print(df.loc[(df['DayOfWeekClaimed']=='0')])
print(df.loc[(df['MonthClaimed']=='0')])

# %%[markdown]
# They are both present in the same column, drop the column!
df2 = df.loc[df['DayOfWeekClaimed']!='0']
df2.reset_index(drop=True, inplace=True)
len(df2)
#Printing the total lenth to check if column has been dropped!
# %%[markdown]
#Age can't be 0. But we have a value which is zero. Checking that out!
len(df2[df2['Age']==0])
#There are 319 rows containg 0 as age!
df2.loc[df2['Age']==0, 'AgeOfPolicyHolder']
#But it also says that all 319 rows Age of Policy holder is between 16-17 years!

# %%[markdown]
df2['Age'].mean
#replacing all with the mean value = 16.5 for easier analysis!
df2.loc[df2['Age']==0,'Age']=16.5

# %%[markdown]
#verifying the result for AGE! 
# print(df2['Age'].unique()==0)
# len(df2[df2['Age']==0])
print(len(df2.drop_duplicates())==len(df2))

# %%[markdown]
# Count the instances of fraud and no fraud
fraud_counts = df2['FraudFound_P'].value_counts()
sns.barplot(x=fraud_counts.index, y=fraud_counts.values)
plt.title('Distribution of Fraudulent and Non-Fraudulent Cases')
plt.xlabel('Fraud Found (0 = No Fraud, 1 = Fraud)')
plt.ylabel('Number of Cases')
plt.xticks(range(2), ['No Fraud', 'Fraud'])
plt.show()
# %%[markdown]
#Univariate analysis
# Identify numerical and categorical columns
numerical_cols = df2.select_dtypes(include=['int64', 'float64']).columns
numerical_cols
categorical_cols = df2.select_dtypes(include=['object', 'category']).columns
categorical_cols

# Plotting histograms for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df2[col], kde=False)
    plt.title(f'Distribution of {col}')
    plt.ylabel('Frequency')
    plt.show()

# Plotting bar charts for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df2)
    plt.title(f'Distribution of {col}')
    plt.show()

# %%[markdown]
#Bivariate plots
# We have already identified the numerical and categorical values. Ploting the graphs for the same.


#%%[markdown]
# Plotting box plots for numerical columns
for col in numerical_cols:
    if col != 'FraudFound_P':  # Exclude the target variable itself
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='FraudFound_P', y=col, data=df2)
        plt.title(f'{col} Distribution by Fraud Found')
        plt.xlabel('Fraud Found (0 = No Fraud, 1 = Fraud)')
        plt.show()

# Plotting grouped bar charts for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='FraudFound_P', data=df2)
    plt.title(f'{col} Distribution by Fraud Found')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Fraud Found', labels=['No Fraud', 'Fraud'])
    plt.show()

# %%[markdown]
# Pie chart for 'FraudFound_P'
fraud_counts = df2['FraudFound_P'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=['No Fraud', 'Fraud'], autopct='%1.1f%%', startangle=140)
plt.title('Fraud Distribution')
plt.show()

# %%[markdown]
sns.lmplot(x='WeekOfMonthClaimed', y='PolicyNumber', hue='FraudFound_P', data=df2, aspect=1.5)
plt.title('Linear Relationship between Two Numerical Features')
plt.show()

#%%
#code to visualize graphs for categorical variables
# List of columns to visualize
columns_to_visualize = ['VehiclePrice', 'AgeOfVehicle', 'Make', 'VehiclePrice']
for col in columns_to_visualize:
    if col in df2.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df2[col], kde=False, bins=20, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
for col in columns_to_visualize:
    if col in df2.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=col, data=df2, palette='pastel')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Frequency')
        plt.show()

#%%
#age distribution graph with density curve
fig, ax = plt.subplots(figsize = (20, 5))

ax.hist(df['Age'], bins = 25, edgecolor = 'black', alpha = 0.7, color = 'skyblue', density = True)
df['Age'].plot(kind = 'kde', color = 'red', ax = ax)

ax.set_xlabel('Age')
ax.set_ylabel('Count / Density')
ax.set_title('Age Distribution Histogram with Density Curve')
ax.legend(['Density Curve', 'Histogram'])
plt.show()

#%%
#pie chart showing the fraud percentage in the dataset
fraud_mapping = {0: 'No', 1: 'Yes'}
df['FraudFound_P_Labels'] = df['FraudFound_P'].map(fraud_mapping)
fraud_counts = df['FraudFound_P_Labels'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of FraudFound_P')
plt.show()

#%%
plt.figure(figsize=(15,8))
sns.countplot(data=df, x="VehiclePrice", hue="FraudFound_P")
contingency_table = pd.crosstab(df['VehiclePrice'], df['FraudFound_P'])
chi2, p_value, _, expected = stats.chi2_contingency(contingency_table)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)
'''
Null Hypothesis (H0): There is no significant association between the price of vehicles and the likelihood of fraud. The proportions of fraudulent claims are the same across different price categories of vehicles.

Alternative Hypothesis (HA): There is a significant association between the price of vehicles and the likelihood of fraud. The proportions of fraudulent claims differ across different price categories of vehicles.

'''

#%%
contingency_table = pd.crosstab(df['WitnessPresent'], df['FraudFound_P'])
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
print("Chi-square statistic:", chi2)
print("p-value:", p_value)
sns.countplot(data=df, x="WitnessPresent", hue="FraudFound_P")


''''
Null Hypothesis (H0): There is no significant association between the presence of a witness and the occurrence of fraud. The proportions of fraudulent claims are the same for claims with and without a witness.

Alternative Hypothesis (HA): There is a significant association between the presence of a witness and the occurrence of fraud. The proportions of fraudulent claims differ between claims with and without a witness.
'''

# %%[markdown]
#Correlation matrix
# Selecting numerical features
numerical_df = df2.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
corr_matrix = numerical_df.corr()
corr_matrix

# %%[markdown]
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix for Numerical Features')
plt.show()


#%%[markdown]
#Based on the correlation matrix:
# WeekOfMonth: -0.011872
# WeekOfMonthClaimed: -0.005783
# Age: -0.026326
# PolicyNumber: -0.020369
# RepNumber: -0.007529
# Deductible: 0.017345
# DriverRating: 0.007259
# Year: -0.024778
##
# The variable with the strongest correlation with 'FraudFound_P' is Age with a coefficient of -0.026326, indicating a weak negative correlation. This is followed by Year with a coefficient of -0.024778.
# %%[markdown]
#Chi test for categorical values
def perform_chi_square_test(df, column, target):
    contingency_table = pd.crosstab(df[column], df[target])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    return chi2, p

# Assuming 'FraudFound_P' is your target variable
target_variable = 'FraudFound_P'

# Looping through categorical columns and performing chi-square tests
for column in df2.select_dtypes(include=['object', 'category']).columns:
    chi2, p = perform_chi_square_test(df2, column, target_variable)
    print(f"Chi-Square Test for {column}:\nChi2 Statistic: {chi2}, P-value: {p}\n")
# %%[markdown]
# Month (Chi2 Statistic: 29.7715, P-value: 0.0017)
# Make (Chi2 Statistic: 59.8153, P-value: 0.000002)
# AccidentArea (Chi2 Statistic: 16.9019, P-value: 0.000039)
# MonthClaimed (Chi2 Statistic: 42.2005, P-value: 0.000015)
# Sex (Chi2 Statistic: 13.4957, P-value: 0.000239)
# Fault (Chi2 Statistic: 264.9846, P-value: 1.4e-59)
# PolicyType (Chi2 Statistic: 437.4914, P-value: 1.8e-89)
# VehicleCategory (Chi2 Statistic: 290.9809, P-value: 6.5e-64)
# VehiclePrice (Chi2 Statistic: 67.8361, P-value: 2.9e-13)
# Days_Policy_Accident (Chi2 Statistic: 11.5698, P-value: 0.0209)
# PastNumberOfClaims (Chi2 Statistic: 53.5418, P-value: 1.4e-11)
# AgeOfVehicle (Chi2 Statistic: 21.9951, P-value: 0.0025)
# AgeOfPolicyHolder (Chi2 Statistic: 33.1049, P-value: 0.000059)
# AgentType (Chi2 Statistic: 7.3805, P-value: 0.0066)
# NumberOfSuppliments (Chi2 Statistic: 18.1555, P-value: 0.000409)
# AddressChange_Claim (Chi2 Statistic: 104.7227, P-value: 9.7e-22)
# BasePolicy (Chi2 Statistic: 402.9472, P-value: 3.2e-88)
##
# Based on the Chi square test, these values have a correlation with the cases being fraud or not.
# %%
# From our EDA and Statistical analysis, we can see that not all the factors that we initially considered are provided in the results. 