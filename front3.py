
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 19:54:33 2022
@author: dgama

"""
import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
import requests
from sklearn.preprocessing import StandardScaler
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')


###############################################################################
############################ 1 SIDEBAR  #######################################
###############################################################################
#Customer ID selection
st.sidebar.header("**Dashboard General Information**")
st.sidebar.markdown("<u>Number of loan applications in the sample :</u>", unsafe_allow_html=True)
st.sidebar.text(10000)

#Pick a number of Applicant data file to analyze
nb_files=st.sidebar.slider("Pick a max number of Applicant data files to analyze", 500,9999 ,500)

num_neighbors=st.sidebar.slider("Pick a number of Applicant files similar to the file being analyzed ",10,15,10)   
###############################################################################
############################ EMBEDDED DATABASE ################################
############################################################################### 
data=pd.read_csv('data.skill.csv').set_index('SK_ID_CURR')
samples=data.iloc[0:nb_files]
target=data.TARGET.value_counts
index_samples = samples.index.values
index_client=samples.index
description=pd.read_csv('appli_descriptif.csv').set_index('Variable')
   
###############################################################################
######################### 2 SIDEBAR ###########################################
###############################################################################


idx_clt = st.sidebar.selectbox("Applicants ID", index_samples)

###############################################################################
data_clt = samples[samples.index == int(idx_clt)]
row_clt=samples.loc[idx_clt].to_list()
###############################################################################
#####SIMILARITY COMPUTING BY KNN EUCLIDEAN DISTANCE METHOD####################♣
###############################################################################
object= StandardScaler()
 

# standardization 
dataset = object.fit_transform(samples) 
print(dataset)

onput=pd.DataFrame(dataset,index=samples.index)
   
rw_clt=onput.loc[idx_clt].to_list()
 
def get_neighbors(samples,rw_clt, num_neighbors):
    
    distances = list()
    for train_row in dataset:
        #calcul distance euclidienne
        distance=0.00
        for i in range(len(train_row)-1):
            distance +=(train_row[i]-rw_clt[i])**2
        dist=sqrt(distance)
        #Fin calcul 
        distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
    ecarts=list()   
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
        ecarts.append(distances[i][1])
    return neighbors,ecarts
     

#let's build list of sigle neighbors std_ dataframe 
neighbors,ecarts = get_neighbors(onput,rw_clt,num_neighbors)
index_list=list()
             
# let's build n+1 neighbors file
index_list=list()
for neighbor in neighbors:
     for(row_index2,row2) in onput.iterrows():
         if(row2==neighbor).all():
             index_list.append(row_index2)
df_index=pd.DataFrame( index=index_list)  

df_index.index = df_index.index.astype(int)
samples.index = samples.index.astype(int)
samples.join(df_index, how='inner')
results=samples.join(df_index, how='inner')
 
ecart_list=list()    
for ecart in ecarts:
      ecart_list.append(ecart)
      
Ranking=range(num_neighbors)
df_ecart=pd.DataFrame(Ranking,columns=['Ranking'])
df_ecart['SK_ID_CURR']=pd.DataFrame(index_list)
df_ecart['Euclidean_distance']=(ecart_list)
df_ecart=df_ecart.set_index('SK_ID_CURR')
Similar_data=pd.concat([df_ecart,results], axis=1,join='inner')

#######END OF SIMILARITY COMPUTING BY KNN EUCLIDEAN DISTANCE METHOD############
###############################################################################
def main() : 
    
    @st.cache(allow_output_mutation=True)
    def load_model():
        pickle_in=open("finalized_model_auc.pkl","rb")
        classifier=pickle.load(pickle_in)
        return classifier
    
        
    @st.cache(allow_output_mutation=True)
    def identite_client(samples, idx_clt):
        data_client = samples[samples.index == int(idx_clt)]
        return data_client
       
    
    @st.cache(allow_output_mutation=True)
    def load_age_population(samples):
        data_age = samples["Age"]
        return data_age
    
    @st.cache(allow_output_mutation=True)
    def load_income_population(samples):
        df_income = pd.DataFrame(samples["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income
                  
        
    @st.cache
    def load_infos_gen(samples):
        lst_infos = [samples.shape[0],round(samples["AMT_INCOME_TOTAL"].mean(),2),round(samples["AMT_CREDIT"].mean(),2)]
    
        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]
        return nb_credits, rev_moy,credits_moy
##############################################################################♥ 
#################### Loading general info######################################
###############################################################################   
    nb_credits, rev_moy,credits_moy = load_infos_gen(data) 
    
###############################################################################
 ########This function sends x to the API and retrieves the prediction#########
###############################################################################    
    @st.cache(allow_output_mutation=True)
    def load_prediction(samples, idx_clt):
       
        url="https://backdefaultprediction.herokuapp.com/info_row"
        headers={
        'Content-Type': 'application/json'}
        X=samples.drop('TARGET',axis=1).loc[idx_clt,:].to_dict()
        fax={'trace':X}
        fax=json.dumps(fax)
        response=requests.post(url,fax)
        proba=response.json()
        prediction=proba['result']
        return prediction
    
    ###########################################################################
    ###################### 3 SIDEBAR DATA #####################################
    ###########################################################################
          
               
    #Average income
    st.sidebar.markdown("<u> Applicants Average income (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)   
    ###########################################################################
    ###################### 4 SIDEBAR DATA #####################################
    ###########################################################################
                 
    #PieChart
    st.sidebar.markdown("<u>Applicants % of Secured and Unsecured loans in the sample </u>", unsafe_allow_html=True)
    series=data['TARGET'].value_counts()
    df_frequence=pd.DataFrame(series)
    size=df_frequence['TARGET']
    fig, ax = plt.subplots(figsize=(4,4))
    label=['Secured loans', 'Unsecured loans']
    plt.pie(size,explode=[0, 0.1],labels=label,autopct='%1.1f%%',startangle=90)
    st.sidebar.pyplot(fig)
    
        
    ###########################################################################
    ####################### HOME PAGE - MAIN CONTENT###########################
    ###########################################################################
    ###########################################################################♦"
    ####################### 1 APPS DISPLAY TITLE############################### 
    ########################################################################### 
     
     
    html_temp1 = """
     <div style="background-color: tomato; padding:10px; border-radius:10px">
     <h1 style="color: white; text-align:center">*****KG_Credit_Bank***** </h1>
     <h2 style="color: white; text-align:center">Dashboard Scoring Credit/Projet_7/V_01</h2>
     <h3 style="color: black; text-align:center">**Credit Risk Project implemented by Didier GAMASSA**</h3>
     <h4 style="color: black; text-align:center">IT_Project_Manager/Data_Scientist</h3>
     <h5 style="color: black; text-align:center">Dev_Support provided by OpenClassrooms/GitHub/Heroku_Dev_Center</h3>
          <h2 style="color: Blue; text-align:center">                  </h2>
     </div>
     <p style="font-size: 20px; font-weight: bold; text-align:center">This Dashboard is a decision tool allowing to validate or not loans to applicants.</p>
     """
    st.markdown(html_temp1, unsafe_allow_html=True)
        
    
    ###########################################################################
    ################ 1 Display Applicant ID from Sidebar########################
    ###########################################################################
   
    st.header("**Applicant ID selection display**")
    st.write("Applicant ID selection :",idx_clt)
    
    
    st.header("**Applicant information display**")
    
    if st.checkbox("Show Applicants information ?"):
        
   ############################################################################
        ############## 2 Applicant information display : #######################
        ############## Applicant Gender, Age, Family status, Children###########
   ############################################################################
          
        infos_client = identite_client(data,idx_clt)
        if infos_client["CODE_GENDER_M"].values[0]==1:
           st.write("**Gender_Male:", "yes")
        else:
           st.write("**Gender_Male:", "no")
        st.write("**Age_(years) : {:.0f} ".format(int(infos_client["Age"])))
        
        if infos_client["NAME_FAMILY_STATUS_Married"].values[0]==0:
            st.write("**Family status_Married : ","no")
        else:
           st.write("**Family status_Maried : ","yes")
                
        st.write("**Number of children : {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
    
        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["Age"]), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("*Income (USD)*")
        st.write("**Income total per year : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Credit amount per year : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Credit as % income per year: **{:.0f}".format(infos_client["Credit_as_percent_income"].values[0]))
        if  infos_client["NAME_INCOME_TYPE_Working"].values[0]==1:
            st.write("**Name income type working :","yes")
        else:
           st.write("**Name income type working :","no")
        
        #Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)
        
        #Relationship Age / Income Total interactive plot 
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(samples, x='Age',y="AMT_INCOME_TOTAL",color='CODE_GENDER_M',
                         hover_data=['AMT_INCOME_TOTAL','NAME_FAMILY_STATUS_Married','CNT_CHILDREN','NAME_EDUCATION_TYPE_Secondarysecondaryspecial'])

        fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Rela tionship Age / Income Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))
    
    
        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'),type="log")
        fig.update_traces(marker={'size': 10})
    
        st.plotly_chart(fig)
    
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
##########################PREDICTION  #########################################
###############   3 Applicant solvability display  #############################
###############################################################################    
    st.header("**Applicant file analysis**")
    prediction = load_prediction(data, idx_clt)
    st.write("The probability of default is {:.2f}%".format(round(prediction*100),2) )  
    
   
    #ıCompute decision according to the best threshold
    if prediction <=.25 :
        decision = "<font color='green'>**Secured loan**</font>" 
    else:
        decision = "<font color='red'>**Unsecured loan :Loan Application Rejected**</font>"
    
    st.write("**Decision** *(with threshold 25%)* **: **", decision, unsafe_allow_html=True)
    
    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, idx_clt)) 

###############################################################################    
################## 4 Feature importance / description #########################
###############################################################################
    if st.checkbox("Applicant ID {:.0f} feature importance ?".format(idx_clt)):
        shap.initjs()
        X=samples.drop('TARGET',axis=1)
        X = X[X.index == idx_clt]
        number = st.slider("Pick a number of features…", 5, 25, 5)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)

 
    if st.checkbox("Need help about Applicant feature description ?") :
           #data,description=load_data()
           list_features = description.index.to_list()
           feature = st.selectbox('Feature checklist…', list_features)
           st.table(description.loc[description.index == feature][:1])
         
    else:
         st.markdown("<i>…</i>", unsafe_allow_html=True)
###############################################################################
###################Display the N Files similar to current File ################
###############################################################################       
    chk_voisins = st.checkbox("Show similar applicant data to current applicant data ?")

    if chk_voisins:
         st.markdown("<u>List of the N(10 to 15) data closest to current applicant data :</u>", unsafe_allow_html=True)
         st.dataframe(Similar_data)
         st.markdown("<i>Ranking 0  = Current Applicant Data</i>", unsafe_allow_html=True)
         st.markdown("<i>Target 1 = Applicant with default</i>", unsafe_allow_html=True)
    else:
         st.markdown("<i>…</i>", unsafe_allow_html=True)
         
         
    st.markdown('***')

if __name__ == '__main__':
    main()

 
