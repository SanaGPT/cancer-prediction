import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from faker import Faker
import random
import torch as t
import torch.nn as n
import torch.optim as o



# Create messy cancer data
fake = Faker()
data = []
for _ in range(50000):
    record = {
        'PatientID': fake.uuid4()[:8] if random.random() > 0.1 else np.nan,
        'Age': random.randint(10, 90) if random.random() > 0.15 else random.choice([999, -1, 'unknown']),
        'Gender': random.choice(['M','F','male','female',np.nan,'Other','X']),
        'TumorSize': f"{random.uniform(0.1, 5.0):.1f}cm" if random.random() > 0.3 else f"{random.randint(1,50)}mm",
        'Diagnosis': random.choice(['Positive','Negative','Malignant','Benign',1,0,'Yes','No']),
        'Smokes': random.choice(['Yes','No',True,False,1,0,'Y','N']),
        'SurvivalDays': max(0, int(np.random.normal(1800, 900))) if random.random() > 0.05 else random.choice([-999,9999])
    }
    data.append(record)

df = pd.DataFrame(data)
df.to_csv('messy_cancer_data.csv', index=False)

#EDA segment 

print(df.head(),'\n',df.tail())
print(df.describe())
print(type(df.columns))
print(df.isnull().value_counts())

#cleaning data1
df.dropna(inplace= True)
df = df.applymap(lambda x: x.lower() if isinstance(x,str) else x)
df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 'yes' if str(x) in 'positive' or  str(x) in 'yes' or  str(x) in 'benign' or  str(x) in 'true' else 'no')
df['Smokes'] = df['Smokes'].apply(lambda x: 'yes' if str(x) == 1 or  str(x) in 'yes' else 'no')
df = df[df['Age'].apply(lambda x: str(x) not in 'unknown')]

def convert_toint(i):
    try:
        if 'cm' in i:
           i = float(i[0:-2]) * 10
           return i
        elif 'mm' in i:
           i = float(i[0:-2])
           return i
    except ValueError:
        print('error')

def gender(letter):
    try:
        if 'f' in letter:
            return 'female'
        elif letter in 'male':
            return 'male'
        else:
            return 'other'
    except ValueError:
        print('error')
    
df['TumorSize'] = df['TumorSize'].apply(convert_toint)
df['Gender'] = df['Gender'].apply(gender)

#storytelling
#how many percentile and how many of patients are suffering cancer
have_cancer = df[df['Diagnosis'] == 'yes'].shape[0]
nocancer =  df[df['Diagnosis'] == 'no'].shape[0]
values = [have_cancer, nocancer]
all = df['Diagnosis'].shape[0]

def percent(whole, clustered):
    return (clustered * 100) / whole

#survive days distribution
'''histplot(df["SurvivalDays"], kde=True)
mat.show()'''

#inspecting if older patients are suffering cancer more
fullhc = df[df['Diagnosis'] == 'yes']
agemean = fullhc['Age'].mean()
print(agemean)

#mow many percentile of patients were male or female
female = percent(all, fullhc[fullhc['Gender'] == 'female'].shape[0])
male = 100 - female
print(f'female:{female}\nmale:{male}')

#the average of tumor size
tumor = fullhc['TumorSize']
tumor_avrage = tumor.mean()

#how many percentile of tomurs are more enormous than average
morethan_average = tumor[tumor > tumor_avrage]
morethan_average_quantity = morethan_average.shape[0]
print(percent(all, morethan_average_quantity))
print(df)

#how many percwntile of patients who smoke suffer cancer
smoker_patient = df[df['Smokes'] == 'yes'].shape[0]
smoker_suffere = fullhc[fullhc['Smokes'] == 'yes'].shape[0]
print(percent(smoker_patient, smoker_suffere))

#preparing data for model
df['Gender'] = df['Gender'].map({'female':1, 'male':0, 'other' : 2})
df['Diagnosis'] = df['Diagnosis'].map({'yes':1, 'no':0})
df['Smokes'] = df['Smokes'].map({'yes':1, 'no':0})
df.dropna(inplace = True)


# Convert all columns to numeric and handle any conversion errors
df[['Smokes', 'TumorSize', 'Gender', 'Age', 'Diagnosis']] = df[['Smokes', 'TumorSize', 'Gender', 'Age', 'Diagnosis']].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace = True)  

label = t.tensor(df['Diagnosis'].values, dtype=t.float32)
feature = t.tensor(df[['Smokes', 'TumorSize', 'Gender', 'Age']].values, dtype=t.float32)

model = n.Sequential(n.Linear(4,4), n.ReLU(), n.Linear(4,1), n.Sigmoid())

#loss & optimizer
criterion = n.BCELoss()
opt = o.SGD(model.parameters(), lr = 0.01)

#trainig model
for epoch in range(200):
    prediction = model(feature).squeeze()
    loss = criterion(prediction, label)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch} loss: {loss.item() : .4f}")

#defining prediction func
def predict(newdata):
    with t.no_grad():
        newdata = t.tensor(newdata, dtype= t.float32)
        output = model(newdata)
        prob = output.item()
        return 'ill' if prob > 0.5 else 'healthy'
    
#creating GUI
def submit():
    try:
        smokese = (1 if 'no' in retort1.get() else 0)
        tumorsizee = int(retort2.get())
        gendere = retort3.get().lower()
        if 'mal' in gendere:
            gendere = 0
        elif 'f' in gendere:
            gendere = 1
        else:
            gendere = 2
        agee = int(retort4.get())

        response = predict([[smokese, tumorsizee, gendere, agee]])
        response = ('patient suffers cancer' if response == 'ill' else 'patient is healthy')
        demonstrate = tk.Label(window, text = response)
        demonstrate.pack(pady = 1)
    except ValueError:
        messagebox.showerror('incorrect input')

window = tk.Tk()
window.title('cancer recognition AI')
window.geometry('600x400')
window.configure(bg = "#4000FF")
custom_font = ('Papyrus', 14)

ques_label1 = tk.Label(window, text = 'does patient smoke?', font = custom_font, bg= "#4000FF", fg ="#FFE600")
ques_label1.pack(pady = 5)
retort1 = tk.Entry(window)
retort1.pack(pady = 2)
ques_label2 = tk.Label(window, text = "what is the tumor size of patient(the unit is mm)", font = custom_font, bg= "#4000FF", fg ="#FFE600")
ques_label2.pack(pady = 5)
retort2 = tk.Entry(window)
retort2.pack(pady = 2)
ques_label3 = tk.Label(window, text = 'what is gender of patient?', font = custom_font, bg= "#4000FF", fg ="#FFE600")
ques_label3.pack(pady = 5)
retort3 = tk.Entry(window)
retort3.pack(pady = 2)
ques_label4 = tk.Label(window, text = 'how old is patient?', font = custom_font, bg= "#4000FF", fg ="#FFE600")
ques_label4.pack(pady = 5)
retort4 = tk.Entry(window)
retort4.pack(pady = 2)
submit_button = tk.Button(window, text= 'confirm', font= custom_font, bg='#FFE600', fg='#4000FF',command = submit)
submit_button.pack(pady = 5)
window.mainloop()
