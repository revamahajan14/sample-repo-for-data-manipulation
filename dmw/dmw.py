import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\ishum\OneDrive\Desktop\data manipulation\dmw\ObesityDataSet_raw_and_data_sinthetic.csv')

print(data.head())
print(data.info())


print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])
data['FAVC'] = le.fit_transform(data['FAVC'])
data['SMOKE'] = le.fit_transform(data['SMOKE'])
data['SCC'] = le.fit_transform(data['SCC'])
data['family_history_with_overweight'] = le.fit_transform(data['family_history_with_overweight'])


data['diet_Risk'] = (
    data['FAVC'] * 2 +
    data['CAEC'].map({'no':0,'Sometimes':1, 'Frequently':2, 'Always':3})
)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


num_cols = [ 'Age' , 'Height','Weight','CH2O','FAF','TUE']

data[num_cols] = scaler.fit_transform(data[num_cols])

data['BMI'] = data['Weight'] / (data['Height'] ** 2)

print(data[['Weight','Height','BMI']].head())

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

data['BMI_Category'] = data['BMI'].apply(bmi_category)

import sqlite3

conn = sqlite3.connect('fitness_warehouse.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE Dim_personal (
    Person_Id INTERGER PRIMARY KEY,
    Age INTEGER,
    Gender INTERGER,
    Height REAL,
    Weight REAL
)
''')
cursor.execute('''
CREATE TABLE Dim_Diet (
    Diet_ID INTERGER PRIMARY KEY,
    FAVC INTEGER,
    FCVC REAL,
    NCP REAL,
    CAEC TEXT
)
''')

cursor.execute('''
CREATE TABLE Dim_Lifestyle (
    Lifestyle_ID INTEGER PRIMARY KEY,
    SMOKE INTEGER,
    SCC INTERGER,
    SH2O REAL,
    FAF REAL,
    TUE REAL
    )
    ''')

cursor.execute('''
CREATE TABLE Dim_Transport (
    Transport_ID INTEGER PRIMARY KEY,
    MTRANS TEXT
)
''')

cursor.execute('''
CREATE TABLE Fact_Health (
    Fact_ID INTEGER PRIMARY KEY,
    Person_ID INTEGER,
    Diet_ID INTEGER,
    Lifestyle_ID INTEGER,
    Transport_ID INTEGER,
    BMI REAL,
    DIET_RISK INTEGER,
    Obesity_Level TEXT,
    FOREIGN KEY(Person_ID) REFERENCES Dim_Personal(Person_ID),
    FOREIGN KEY(DIET_ID) REFERENCES Dim_Diet(Diet_ID),
    FOREIGN KEY(Lifestyle_ID) REFERENCES Dim_Lifestyle(Lifestyle_ID),
    FOREIGN KEY(Transport_ID) REFERENCES Dim_Transport(Transport_ID)
)
''')

data['Person_ID'] = range(1, len(data)+1)
data['Diet_ID'] = range(1, len(data)+1)
data['Lifestyle_ID'] = range(1 ,len(data)+1)
data['Transport_ID'] = range(1 , len(data)+1)

for _, row in data.iterrows():
    cursor.execute('''
    INSERT INTO Dim_Personal VALUES (?,?,?,?,?)
    ''', (row['Person_ID'],row['Age'],row['Gender'],row['Height'],row['Weight']))