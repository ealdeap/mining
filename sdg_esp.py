import altair as alt
from altair.vegalite.v4.schema.channels import Color, Order
from altair.vegalite.v4.schema.core import Align, Baseline
from attr import define
from matplotlib import scale
from numpy.testing._private.utils import print_assert_equal
from pandas.io import excel
from pandas.tseries.offsets import BQuarterBegin
from seaborn.rcmod import set_theme
import streamlit as st
from streamlit.elements.arrow_altair import ChartType
import sys
import os
import re
import csv
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pickle
from PIL import Image
import plotly.graph_objects as go
from urllib.error import URLError
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from IPython.display import Markdown, display
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


nltk.download('stopwords')


st.set_page_config(layout="wide")


## load data 
@st.cache
def load_data_raw(nrows):
    data = pd.read_excel("scopus_final_es_ml.xlsx", index_col = 0, nrows=nrows)
    return data
    #test_data = pd.read_excel("test_data.xlsx")

def load_data_chart(nrows):
    data = pd.read_excel("chart.xlsx", nrows=nrows)
    return data

def load_data_sample(nrows):
    data = pd.read_excel("Sample.xlsx", index_col=0, nrows=nrows)
    return data

def load_data_accuracy(nrows):
    data = pd.read_excel("accuracy.xlsx", nrows=nrows)
    return data

# load datasets
data_raw = load_data_raw(15000)
data_chart = load_data_chart(16)
sample_data = load_data_sample(10)
accuracy = load_data_accuracy(20)



st.sidebar.header("Bienvenido al Clasificador de ODS")

st.sidebar.image("SDG.png")


# Create a page dropdown 
page = st.sidebar.selectbox("Seleccion una Página", ["Información de Muestra", "Clasificar un Texto", "Clasificar un Archivo Excel"]) 




if page == "Información de Muestra":

    st.header("Información de Muestra de los ODS")

    st.subheader("En esta sección encontrarás información relevante acerca de la muestra de artículos y publicaciones cientificas que estamos usando como referencia para la clasificación de textos")
    #if st.checkbox("Show SDG data"):
        #st.subheader("SDG Sample Data")

    st.write("")

    st.write("La base de datos de artículos usados como referencia para la clasificación de los ODS está en constante crecimiento, ya contamos con mas de 11.000 artículos incluidos y una efectividad promedio en la clasificación de los textos superior al 92%")
    
    st.write("")

    col0, col1, col2, col3 = st.columns(4)

    col1.metric("Efectividad Promedio: ", "93,2%", delta= "3,2 puntos superior")
    col2.metric("Total de Artículos: ", 11.195, delta= "8.071 articulos nuevos")

    st.write("")
    st.write("")


    st.subheader("Efectividad individual de clasificación de los ODS")

    #### radar chart


    r = accuracy["accuracy"]
    theta = accuracy["Objetivo"]

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = r,
        theta = theta,
        fill='toself',
        name='ODS efectividad de clasificación'
    ))
    

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    ##### bar chart

    st.write("")
    st.subheader("Cantidad de textos clasificados por cada ODS")
    st.write("")
    st.write("La cantidad de textos que han sido clasificados por cada ODS hasta el momento se muestran en el siguiente gráfico:")
    st.write("")
    

    sns.set(font_scale = 1)

    categories = pd.DataFrame(data_raw.columns.values)[1:18]
    categories = categories.reset_index(drop = True)

    val = pd.DataFrame(data_raw.iloc[:,1:18].sum())
    val = val.reset_index(drop = True)

    p = pd.concat([categories, val], axis = 1)

    p = p.set_axis(['category', 'value'], axis=1, inplace=False)

    gra, ax = plt.subplots(figsize=(15, 10)) 

    ax = sns.barplot(p["category"], p["value"], color="m")

    rects = ax.patches
    labels = p["value"]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=8)

    plt.ylabel('Cantidad de Artículos', fontsize=15)
    plt.xlabel('Número del ODS', fontsize=15)
    

    st.pyplot(gra)
    
    #c = alt.Chart(p).mark_bar().encode(alt.X("category", 
    #sort= ["goal_1" ,"goal_2" ,"goal_3" ,"goal_4" ,"goal_5" ,"goal_6" ,
    #"goal_7" ,"goal_8" ,"goal_9" ,"goal_10","goal_11","goal_12","goal_13",
    #"goal_14","goal_15","goal_16"]), 
    #y = "value").properties(width = "container", height = 600)


    #st.altair_chart(c, use_container_width=True)



    def color_SDG(val):
        if val == "Yes":
            color = 'green'
        else:
            color = ''
        return f'background-color: {color}'

    st.subheader("Ejemplo del resultado al clasificar una serie de textos")
    st.write("")
    st.write("Al clasificar un archivo excel que contenga la serie de textos a clasificar, esta herramienta arrojará como resultado una tabla que contendrá 'Yes' o 'No' como indicador de si es que el texto ha logrado ser mapeado a alguno de los ODS, como se muestra en el siguiente ejemplo:")
    st.write("")
    st.dataframe(sample_data.T.style.applymap(color_SDG), height= 3000)





if page == "Clasificar un Texto":
    
    st.title("Clasificar un Texto en Relación a los ODS")

    st.subheader("Utilizando un Párrafo")

    #  input for the model

    xtest = st.text_area("Clasificar", "Clasifica un texto asociado a los ODS")

    new = {"text": [xtest]}

    test_data = pd.DataFrame(new)

    # model itself

    categories = list(data_raw.columns.values)[1:17]
    print(categories)

    data = data_raw

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    def cleanHtml(sentence):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext

    def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned

    def keepAlpha(sentence):
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    test_data['text'] = test_data['text'].str.lower()
    test_data['text'] = test_data['text'].apply(cleanHtml)
    test_data['text'] = test_data['text'].apply(cleanPunc)
    test_data['text'] = test_data['text'].apply(keepAlpha)

    #Removing stop words

    stop_words = set(stopwords.words('spanish'))
    stop_words.update(['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez', 'poder', 'tambien', 'hasta', 'entre', 'junto', 'sin', 'embargo', 'todavía', 'dentro', 'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'durante', 'para', 'por', 'sin', 'segun', 'sobre', 'tras'])

    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    def removeStopWords(sentence):
            global re_stop_words
            return re_stop_words.sub(" ", sentence)
    test_data['text'] = test_data['text'].apply(removeStopWords)


    #Stemming

    stemmer = SnowballStemmer("spanish")
    def stemming(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    test_data['text'] = test_data['text'].apply(stemming)
    test_data['text'] = test_data['text'].str.lower()
    test_data['text'] = test_data['text'].apply(cleanHtml)
    test_data['text'] = test_data['text'].apply(cleanPunc)
    test_data['text'] = test_data['text'].apply(keepAlpha)

    # test and train data partitioning...

    original_test_data = test_data
    test = test_data
    print(test.shape)

    test_text = test['text']
    print("test")
    print(test_text)

    # Importing Pickle

    pickle_in = open("tf_idf_vectorizer.pickle","rb")
    vectorizer = pickle.load(pickle_in)
    print(len(vectorizer.get_feature_names()))

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['text'], axis=1)

    #Multiple Binary Classifications - (One Vs Rest Classifier)

    def printmd(string):
        display(Markdown(string))

    # Using pipeline for applying logistic regression and one vs rest classifier

    LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
                ])

    arrs = []

    # Importing Pickle

    pickle_in = open("trained_model_esp.pickle","rb")
    pipeline_array = pickle.load(pickle_in)

    # Generating predictions

    for index in range(0,len(categories)):
        printmd('**Processing {} review...**'.format(categories[index]))
        LogReg_pipeline = pickle.loads(pipeline_array[index])
        prediction = LogReg_pipeline.predict(x_test)
        arrs.append(prediction)
        print("Prediction: ")
        print(prediction)
        print("\n")

    # Generating result vector

    output_array = []
    output_array.append(["text", "goal_1" ,"goal_2" ,"goal_3" ,"goal_4" ,"goal_5" ,"goal_6" ,
        "goal_7" ,"goal_8" ,"goal_9" ,"goal_10","goal_11","goal_12",
        "goal_13","goal_14","goal_15","goal_16"])

    test_review = original_test_data["text"].values

    for index in range(0,len(test_review)):
        row = []
        row.append(test_review[index])
        for arr in arrs:
            row.append(arr[index])
        output_array.append(row)

    result = pd.DataFrame(output_array)


    # Paragraph Classifier

    if st.button("Clasifica el texto"):
        st.success("Los ODS relacionados con tu texto son:")
        st.write(result.set_index(result.columns[0]).T, use_container_width=True)




# Model for excel Files
if page == "Clasificar un Archivo Excel":

    # Model for excel Files

        uploaded_file = st.file_uploader(label="upload here", type="xlsx")

        test_data = pd.read_excel("esp.xlsx")

        if uploaded_file:
            test_data = pd.read_excel(uploaded_file)
                #st.dataframe(df)
                #st.table(df)


        data_raw = pd.read_excel("scopus_final_es_ml.xlsx")
        #test_data = pd.read_excel("SDG_ml.xlsx")

        # test_data = pd.DataFrame(new)
        print("**Sample data:**")
        test_data.head()

        categories = list(data_raw.columns.values)[1:18]
        print(categories)

        #Data Pre-Processing

        import nltk
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        import re
        import sys
        import warnings


        data = data_raw

        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        def cleanHtml(sentence):
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, ' ', str(sentence))
            return cleantext
        def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
            cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
            cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
            cleaned = cleaned.strip()
            cleaned = cleaned.replace("\n"," ")
            return cleaned
        def keepAlpha(sentence):
            alpha_sent = ""
            for word in sentence.split():
                alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
                alpha_sent += alpha_word
                alpha_sent += " "
            alpha_sent = alpha_sent.strip()
            return alpha_sent

        test_data['text'] = test_data['text'].str.lower()
        test_data['text'] = test_data['text'].apply(cleanHtml)
        test_data['text'] = test_data['text'].apply(cleanPunc)
        test_data['text'] = test_data['text'].apply(keepAlpha)

        #Removing stop words
        stop_words = set(stopwords.words('spanish'))
        stop_words.update(['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez', 'poder', 'tambien', 'hasta', 'entre', 'junto', 'sin', 'embargo', 'todavía', 'dentro', 'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'durante', 'para', 'por', 'sin', 'segun', 'sobre', 'tras'])

        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
        def removeStopWords(sentence):
            global re_stop_words
            return re_stop_words.sub(" ", sentence)
        test_data['text'] = test_data['text'].apply(removeStopWords)


        #Stemming
        stemmer = SnowballStemmer("spanish")
        def stemming(sentence):
            stemSentence = ""
            for word in sentence.split():
                stem = stemmer.stem(word)
                stemSentence += stem
                stemSentence += " "
            stemSentence = stemSentence.strip()
            return stemSentence
        test_data['text'] = test_data['text'].apply(stemming)

        test_data['text'] = test_data['text'].str.lower()
        test_data['text'] = test_data['text'].apply(cleanHtml)
        test_data['text'] = test_data['text'].apply(cleanPunc)
        test_data['text'] = test_data['text'].apply(keepAlpha)



        # test and train data partitioning...

        from sklearn.model_selection import train_test_split
        original_test_data = test_data
        test = test_data
        print(test.shape)

        test_text = test['text']
        print("test")
        print(test_text)


        from sklearn.feature_extraction.text import TfidfVectorizer
        import pickle

        pickle_in = open("tf_idf_vectorizer.pickle","rb")
        vectorizer = pickle.load(pickle_in)
        print(len(vectorizer.get_feature_names()))

        x_test = vectorizer.transform(test_text)
        y_test = test.drop(labels = ['text'], axis=1)


        #Multiple Binary Classifications - (One Vs Rest Classifier)

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score
        from sklearn.multiclass import OneVsRestClassifier
        from IPython.display import Markdown, display
        import pickle
        def printmd(string):
            display(Markdown(string))


        # Using pipeline for applying logistic regression and one vs rest classifier
        LogReg_pipeline = Pipeline([
                        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
                    ])

        arrs = []

        pickle_in = open("trained_model_esp.pickle","rb")
        pipeline_array = pickle.load(pickle_in)

        for index in range(0,len(categories)):
            printmd('**Processing {} review...**'.format(categories[index]))
            LogReg_pipeline = pickle.loads(pipeline_array[index])
            prediction = LogReg_pipeline.predict(x_test)
            arrs.append(prediction)
            print("Prediction: ")
            print(prediction)
            #print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
            print("\n")

        output_array = []
        output_array.append(["text", "goal_1" ,"goal_2" ,"goal_3" ,"goal_4" ,"goal_5" ,"goal_6" ,
            "goal_7" ,"goal_8" ,"goal_9" ,"goal_10","goal_11","goal_12",
            "goal_13","goal_14","goal_15","goal_16", "goal_17"])

        test_review = original_test_data["text"].values

        for index in range(0,len(test_review)):
            row = []
            row.append(test_review[index])
            for arr in arrs:
                row.append(arr[index])
            output_array.append(row)

        result = pd.DataFrame(output_array)
        result.columns = result.iloc[0]
        result = result[1:]

        result['sum'] = result.iloc[:,1:18].sum(axis = 1)
        result = result[result["sum"]!= 0]
        result = result.sort_values(by = ["sum"], ascending = False)

        def color_SDG(val):
            if val == 1:
                color = 'green'
            else:
                color = ''
            return f'background-color: {color}'

        # Paragraph Classifier

        if st.button("Classify SDG"):
            st.success("The SDG related to the text are:")
            st.write(result.style.applymap(color_SDG))


