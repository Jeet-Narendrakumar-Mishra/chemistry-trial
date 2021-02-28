#Imported Pandas To Read The CSV File
import pandas as pd
#Imported Streamlit To Make The App USER_FRIENDLY
import streamlit as st
#Imported sklearn For Machine Learning Models and Training the Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#Imported Chemparse To Break The Atoms In Individual Molecules
import chemparse

#Writing The Header Of The App
st.write("""
# Crystal System Prediction
To predict the crystal system class of the battery
""")

#Reading the dataset
data = pd.read_csv("lithium-ion_batteries.csv")

#Using The Chemparse module to parse the forumla column. We basically want each molecule to be split into columns of each atom with values being the amount of atoms in the particular molecule.
chem_data = data.Formula.apply(chemparse.parse_formula)
chem_data = pd.json_normalize(chem_data)
chem_data= chem_data.fillna(0)

#Adding chem_data to original DataFrame
data = data.join(chem_data)

#Modifying Data According To The Needs
data = data.drop(columns = ["Materials Id"])

#Writing The Data Into The App
st.subheader("Data Information")
st.dataframe(data)
st.write(data.describe())

#Displaying The Bar Graph
chart = st.bar_chart(data)

#Taking User Input
def get_user_input():
    #No. Of Atoms Chosen By User
    li = st.sidebar.slider("Number Of Lithium Atoms",0,20,2)
    mn = st.sidebar.slider("Number Of Manganese Atoms",0,20,1)
    si = st.sidebar.slider("Number Of Silicon Atoms",0,20,2)
    o = st.sidebar.slider("Number Of Oxygen Atoms",0,50,8)
    fe = st.sidebar.slider("Number Of Iron Atoms",0,10,1)
    co = st.sidebar.slider("Number Of Cobalt Atoms",0,10,0)
    formation_energy = st.sidebar.slider("Formation Energy In eV",-3.0,0.0,-2.61)
    e_above_hull = st.sidebar.slider("Energy Above Hull in eV",0.0,0.2,0.0582)
    band_gap = st.sidebar.slider("Band Gap In eV",0.0,4.0,2.07)
    nsites = st.sidebar.slider("No. Of Atoms In Unit Cell Per Crystal",0,135,39)
    density = st.sidebar.slider("Density in gm/cc",2.0,4.5,2.98)
    volume = st.sidebar.slider("The unit cell volume of the material",120.0,1520.0,467.7)
    bandstructure = st.sidebar.slider("Has Bandstructure",0,1,0)
    user_data = {"Formation Energy (eV)": formation_energy,"E Above Hull": e_above_hull,"Band Gap (eV)": band_gap,"Nsites": nsites,"Density (gm/cc)": density,"Volume": volume,"Has Bandstructure": bandstructure,"Li" : li, "Mn" : mn, "Si" : si, "O" : o,"Fe" : fe, "Co" : co}
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

#Splitting the data into feature and labels
y = data["Crystal System"]
X = data.drop(columns = ["Crystal System", "Formula", "Spacegroup"])

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

predictions = model.predict(user_input)

st.subheader("Predictions")
st.write(predictions)



