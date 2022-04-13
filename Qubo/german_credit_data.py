#!/usr/local/bin/python3

import pandas as pd

def german_credit_data ():
    try:
        #first transformed german.data into german.csv
        column_Names = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]
        #column_Names = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for','Telephone','foreign worker', 'Good/Bad credit]
        dataframe = pd.read_csv("../Data_folder/German/german.csv", names=column_Names)
    except:
        print("Import dataframe error")
    return dataframe