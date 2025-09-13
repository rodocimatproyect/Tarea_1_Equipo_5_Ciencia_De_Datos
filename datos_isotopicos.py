# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

# Dependencias

import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np


import statsmodels.api as sm
from scipy import stats
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import scipy.stats as st

# Configuraciones del IDE
pd.set_option("display.max_columns", 8)  # Mostrar todas las columnas
pd.set_option("display.max_rows", 5)     # Mostrar todas las filas
pd.set_option("display.width", 120)        # Ajuste automático al ancho de la terminal
pd.set_option("display.max_colwidth", None) # No truncar el contenido de celdas
pd.set_option("display.expand_frame_repr", False)



# Ruta de Trabajo y Datos
os.chdir(r"C:\Users\ramir\OneDrive\Documentos\CIMAT\tercer_semestre\Ciencia de Datos")


# DEfinimos el nombre del txt
pathdata = r'.\2023-002_ISONET-Project-Members_13C_Data.txt'
#cargamos la base de datos
df = pd.read_csv(pathdata, encoding="latin1",
                 skiprows=[0,1,2],sep = '\t', dtype=str, header = None)
# tratamiento previo de la tabla, la transponemos para que se le puede¿a hacer 
# one hot encoding a las variables dummies
df = df.T 
df.columns =  df.loc[[0]].values[0].tolist()
df = df.drop([0])

# Casteamos los datos, para los casos que son numéricos o Strings
string =  ['Site Code', 'Site  name', 'Country', 'Latitude', 'Longitude',
       'Species', 'First year CE', 'Last year CE', 'elevation a.s.l.',
       'Year CE']
numeric = [i for i in df.columns if i not in string ]
df.columns = string + numeric
df[string] = df[string].astype(str)
df[numeric] = df[numeric].replace(",", ".", regex=True)
df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
df['Site Code'] = df['Site Code'].apply(lambda x: "".join(x.split()))

# ===================   Detección de Problemas en los Datos =================================
# Porcentaje de Faltantes por año
percmissingyear = (df[numeric].isnull().sum(axis = 0)/df.shape[0])
# porcentajed de faltantes por sitio
percmissingcodesite = df[['Site Code'] + numeric].set_index('Site Code').isnull().sum(axis = 1)/df[numeric].shape[1]

#============================================================================================

# A coontinuación cada una de las funciones recibe como argumentos 
# las variables df que corresponde al data frame con los datos 
# y site_key, que se refiere al código del sitio que queremos anlizar 

# Graficación de La serie de tiempo 
def time_series(df, site_key):
    df[df['Site Code'] == site_key][numeric].T.plot()
    plt.title(site_key)
    plt.xlabel("Año")
    plt.ylabel("Mediciones")
    
# Gráfico de puntos para una serie específica
def scatterplot(df, site_key):
    df1 = df[df['Site Code'] == site_key][numeric].T.reset_index()
    df1.columns = ['site', 'values']
    plt.scatter(df1["site"].astype(int), df1['values'], color = 'white', edgecolors= 'black')
    plt.suptitle(site_key, fontsize=16, fontweight="bold")
    plt.xlabel("Año")
    plt.ylabel("Medición")
    plt.show()
    
# Función para diferenciar la serie
def calcula_diferencias(df):
    return df.diff()

# Función para calcular el z-score
def z_score(x, media, desviacion_estandar):
    return (x - media) / desviacion_estandar

# Función que marca outliers mediante el método de diferencias 
def marca_outliers(df, site_key, method = 'IQR'):
    df1 = calcula_diferencias(df[df['Site Code'] == site_key][numeric].T).dropna()
    df1.columns = ['values']
    dfdata =  df1.values
    dts_sn_na = pd.Series([i[0] for i in dfdata])
    # Calcular cuartiles
    if method == 'IQR':
        Q1 = dts_sn_na.quantile(0.25)
        Q3 = dts_sn_na.quantile(0.75)
        IQR = Q3 - Q1

        # Limites para detectar outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        index_outliers = df1[(df1['values'] < lower_bound) | (df1['values'] > upper_bound)].index.to_list()
        df1['outlier'] = [1 if j in index_outliers else 0 for j in df1.index.values]
    else:
        media = dts_sn_na.mean()
        std =  np.sqrt(dts_sn_na.var())
        df1['outlier'] = df1['values'].apply(lambda x:1 if abs(z_score(x, media, std))>3 else 0)
    return df1
    

# Función que grafica outliers mediante el método de diferencias 
def grafica_outliers(df, site_key, method = 'IQR'):
    
    df1 = marca_outliers(df, site_key, method)
    num_outliers = df1[df1['outlier']==1].shape[0]
    index_outliers = list(df1[df1['outlier']==1].index)
    df1 = df1.reset_index()
    df1['outlier'] = df1['outlier'].apply(lambda val: 'red' if val == 1 else 'white')
    df_orig = df[df['Site Code'] == site_key][numeric].T
    df_orig.columns = ['values']
    df_orig['outlier'] = [1 if j in index_outliers else 0 for j in df_orig.index.values]
    print(site_key , '---', df_orig['outlier'].sum())
    df_orig['outlier'] = df_orig['outlier'].apply(lambda val: 'red' if val == 1 else 'white')
    
    
    fig, axes = plt.subplots(1, 2, figsize=(12,5))  # 1 fila, 2 columnas

    # --- Gráfica 1: outliers diferencias ---
    axes[0].scatter(df1["index"].astype(int), df1["values"], color=df1["outlier"], edgecolors="black")
    axes[0].set_title("Outliers diferencias")
    axes[0].set_xlabel("Año")
    axes[0].set_ylabel("Medición")
    
    # --- Gráfica 2: outliers datos originales ---
    axes[1].scatter(df_orig.index.astype(int), df_orig["values"], color=df_orig["outlier"], edgecolors="black")
    axes[1].set_title("Outliers datos originales")
    axes[1].set_xlabel("Año")
    axes[1].set_ylabel("Medición")
    
    fig.suptitle(site_key, fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.show()
    
    return num_outliers



# Función para detección de outliers mediante el método de regresión 
def outliers_detection_regression(df, site_key, plotit = 0):
    dfhola = df[df['Site Code'] == site_key][numeric].T.reset_index()
    dfhola.columns = ['time', 'values']
    dfhola = dfhola.dropna().reset_index(drop = True)
    X = sm.add_constant(dfhola[['time']])
    y = np.array(dfhola['values'])
    t = dfhola['time']
    model = sm.OLS(np.array(dfhola['values']),X.astype(float) ).fit()
    #print(model.summary())
    
    fitted = model.fittedvalues
    resid = model.resid
    #print(model.params)
    beta0 = model.params.values[0]
    beta1 = model.params.values[1]
    x_line = np.array([int(t.min()), int(t.max())])
    y_line = beta1 * x_line + beta0
    #print('hola')
    mean = resid.mean()
    dvs = resid.std()
    dfhola['outlier'] = np.where(np.abs(resid)> mean + 3*dvs, 1, 0)
    print(site_key , '---', dfhola['outlier'].sum())
    
    dfhola['outlier'] = ['red' if i==1 else 'white' for i in dfhola['outlier']]
    
    if plotit == 1:
    
        fig, axes = plt.subplots(1, 2, figsize=(12,5))  # 1 fila, 2 columnas
        
        # --- Gráfica 1: errores ---
        axes[0].scatter(fitted, resid, color=dfhola['outlier'], edgecolors="black")
        axes[0].axhline(y=mean + 3*dvs, color='red',linestyle="--", label="+3σ")
        axes[0].axhline(y=mean - 3*dvs, color='red',linestyle="--", label="-3σ")
        axes[0].set_title("Errores")
        axes[0].set_xlabel("Fitted")
        axes[0].set_ylabel("Errors")
        
        # --- Gráfica 2: original ---
        axes[1].scatter(t.astype(int), y, color=dfhola["outlier"], edgecolors="black")
        axes[1].set_title("Outliers datos originales")
        axes[1].set_xlabel("Año")
        axes[1].set_ylabel("Medición")
        axes[1].plot(x_line, y_line, color='red', linewidth=2, linestyle="--")
        
        fig.suptitle(site_key, fontsize=16, fontweight="bold")
        
        plt.tight_layout()
        plt.show()
    
    return dfhola[dfhola['outlier']=='red']
    
    

# Función para imputar missings mediante el ajuste de una recta a los datos
def imputa_missings(df, site_key):
    dfhola = df[df['Site Code'] == site_key][numeric].T.reset_index()
    dfhola.columns = ['time', 'values']

    first_date = dfhola[dfhola['values'].isnull()==False]['time'].astype(int).min()
    last_date = dfhola[dfhola['values'].isnull()==False]['time'].astype(int).max()
    rango_fecha = [str(i) for i in range(first_date, last_date+1)]
    df_filtrado = dfhola[dfhola['time'].isin(rango_fecha)]
    num_missings = df_filtrado['values'].isnull().sum()
    #print(site_key, '--', num_missings)

    #quita outliers
    outliers_info = outliers_detection_regression(df, site_key)
    dates_outliers = outliers_info['time'].to_list()
    df_noout = df_filtrado[[not i for i in df_filtrado['time'].isin(dates_outliers)]]

    # Ajusta Regresión
    df_noout1 = df_noout.dropna().reset_index(drop = True)
    n = df_noout1.shape[0]
    datesmissing = df_noout[df_noout['values'].isnull()==True]['time'].astype(int).to_list()
    X = sm.add_constant(df_noout1[['time']])
    y = np.array(df_noout1['values'])
    t = df_noout1['time']
    model = sm.OLS(y,X.astype(float) ).fit()


    residuos = model.resid

    beta0 = model.params.values[0]
    beta1 = model.params.values[1]

    #simula predicción
    x_mean = df_noout1['time'].astype(int).mean()
    #print(x_mean)
    x_line = np.array(datesmissing) #puntos que queremos predecir
    y_line = beta1 * x_line + beta0
    #print(y_line)

    SSE = np.sum(residuos**2)                 
    MSRes = SSE / (n - 2) 
       
    Sxx = np.sum((df_noout1['time'].astype(int) - x_mean)**2)
    Var_pred = MSRes * (1 + 1/n + (x_line - x_mean)**2 / Sxx)
    SE_pred = np.sqrt(Var_pred)
    #print(SE_pred)

    t_sample = st.t.rvs(df=n-2, size=len(datesmissing))   # un valor t
    #print(t_sample)
    y_sim = y_line + SE_pred * t_sample

    predictions = pd.DataFrame(zip([str(i) for i in datesmissing], y_sim), columns= ['time', 'values_sim'])
    df_hola2 = df_noout.merge(predictions, on = 'time', how = 'left')
    df_hola2['missing'] = np.where(df_hola2['values'].isnull() == True, 1, 0)
    df_hola2['values_f'] = df_hola2['values'].fillna(df_hola2['values_sim'])
    df_hola2 = df_hola2.drop(columns = ['values', 'values_sim'])
    df_hola2.columns = ['time', 'missing', 'values']



    # Regresamos los outliers
    df_outliers = outliers_info[['time', 'values']]
    df_outliers['missing'] = 0
    df_final = pd.concat([df_hola2, df_outliers]).sort_values(['time'])

    #grafica

    df_final['missing'] = ['blue' if i==1 else 'white' for i in df_final['missing']]

    plt.scatter(df_final['time'].astype(int), df_final['values'],
                color=df_final['missing'], edgecolors="black")
    plt.title(site_key)
    plt.xlabel("Año")
    plt.ylabel("Mediciones")
    plt.show()
    





#===========================================   Salida de la consola ======================================
print('======================== Resumen del Proyecto ============================')
print('---------- Data Frame ----------')
display(df)

print('======================= Series de Tiempo de los sitios ===================')
for i in df['Site Code']:
    time_series(df, i)
    
print('======================= Prueba de Shapiro Wilks =======================')
#queremos ver la normalidad de los errores
for i in df['Site Code'].to_list():
    dfhola = df[df['Site Code'] == i][numeric].T.reset_index()
    dfhola.columns = ['time', 'values']
    dfhola = dfhola.dropna().reset_index(drop = True)

    X = sm.add_constant(dfhola[['time']])
    y = np.array(dfhola['values'])
    t = dfhola['time']
    model = sm.OLS(np.array(dfhola['values']),X.astype(float) ).fit()
    #print(model.summary())

    fitted = model.fittedvalues
    resid = model.resid

    shapiro_stat, shapiro_p = stats.shapiro(resid)
    print(i, '   -----    ', shapiro_p)

print('======================= Grafica_outliers mediante el método de Diferencias =======================')
print('Nombre de Sitio ---- Número de Outliers')
for i in df['Site Code']:
    grafica_outliers(df, i)

print('======================= Grafica_outliers mediante el método de Ajuste de Regresión =======================')
print('Nombre de Sitio ---- Número de Outliers')
for i in df['Site Code']:
    outliers_detection_regression(df, i, 1)

print('======================= Imputación de Missings  =======================')
print('Nombre de Sitio ---- Número de Missings')
for i in df['Site Code']:
    imputa_missings(df, i)

print('=============')
df_T=df.T
df=df.T
min_max=df.iloc[10:].copy()
min_max_scaler = MinMaxScaler()
cadena_fechas = np.zeros(406)
for i in range(406):
    cadena_fechas[i]=1600+i
df_fechas_normal = pd.DataFrame(cadena_fechas, columns=['original_data'])

#definicion de funcion min-max
def funcion_min_max(df,i):
    df[i] = pd.to_numeric(df[i], errors='coerce')
    mediana_col1 = df[i].median()
    # Rellenar los valores NaN con la mediana
    df[i] = df[i].fillna(mediana_col1)
    #"DataFrame después de reemplazar 'nan' por la mediana:"
    df[26] = min_max_scaler.fit_transform(df[[i]])
    return df

#Min-max de las columnas 23-24 de min_max
columnas=[21,24,17]
df_fechas=funcion_min_max(df_fechas_normal,'original_data')

plt.figure(figsize=(8, 6))

for i in range(len(columnas)):
    nuevo_df=funcion_min_max(min_max,columnas[i])
    plt.scatter(df_fechas[26], nuevo_df[26], color='purple', marker='o', s=10)
    # Añadir etiquetas y título
    plt.xlabel('Años')
    plt.ylabel('Isopos')
    plt.title('Transformacion Min-Max')

    # Mostrar el gráfico
    plt.grid(True)
    plt.show()

#___________________________________z-score_______________________________________
z_score_scaler = StandardScaler()

def funcion_z_score(df,i):
    df[i] = pd.to_numeric(df[i], errors='coerce')
    mediana_col1 = df[i].median()
    # Rellenar los valores NaN con la mediana
    df[i] = df[i].fillna(mediana_col1)
    #"DataFrame después de reemplazar 'nan' por la mediana:"
    df[25] = z_score_scaler.fit_transform(df[[i]])
    return df

plt.figure(figsize=(8, 6))

for i in range(len(columnas)):
    nuevo_df=funcion_z_score(min_max,columnas[i])
    plt.scatter(df_fechas[26], nuevo_df[25], color='purple', marker='o', s=10)
    # Añadir etiquetas y título
    plt.xlabel('Años')
    plt.ylabel('Isopos')
    plt.title('Transformacion z-score')

    # Mostrar el gráfico
    plt.grid(True)
    plt.show()

#___________________Histogramas
columnas_his=[17,21,24]
media=np.zeros(len(columnas_his))
desvicacion=np.zeros(len(columnas_his))
c=0
for i in columnas_his:
    hist=min_max[i].hist(bins=9) #Numero de columnas que queremos en el histograma
    media[c]= min_max[i].mean()
    desvicacion[c]=min_max[i].std()
    c+=1
    plt.xlabel("Isopos"+str(i))
    plt.show()

print(media)
print(desvicacion)
#__________________________Plot de datos de isopos y años


df_nuevo=df_T.iloc[10:].copy()

def plot_disp(df_fechas_normal,df_nuevo,i):
    
    plt.scatter(df_fechas_normal['original_data'], df_nuevo[i], alpha=0.7)
    plt.xlabel("Años")
    plt.ylabel("Isopos")
    plt.title("Gráfico de dispersión")
    plt.show()

    df_nuevo[25] = df_nuevo[i].fillna(df_nuevo[i].mean())
    df_nuevo[26] = df_nuevo[i].isna()
    plt.scatter(df_fechas_normal['original_data'], df_nuevo[25], c=df_nuevo[26].map({True: "red", False: "blue"}), 
                alpha=0.7)
    plt.xlabel("Años")
    plt.ylabel("Isopos")
    plt.title("Gráfico de dispersión")
    plt.show()
    return

for i in [17,21,24]:
    plot_disp(df_fechas_normal,df_nuevo,i)



#_________________Precencia de Outliers________________
df_nuevo_4=df_T.iloc[10:].copy()
def detect_outliers_IQR(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)

def plot_detect_outliers_IQR_year(df_nan,i,df_no_nan,j,f):
    x = pd.to_numeric(df_nan[i].reset_index(drop=True), errors="coerce")
    y = pd.to_numeric(df_no_nan[j].reset_index(drop=True), errors="coerce")
    df = pd.DataFrame({"x": x, "y": y})
    print(df)
    outliers_x = f(df["x"])
    outliers_y = f(df["y"])

    # Consideramos un punto outlier si X o Y es outlier
    df["is_outlier"] = outliers_x | outliers_y

    # --- Scatter plot ---
    plt.scatter( df.loc[~df["is_outlier"], "y"], df.loc[~df["is_outlier"], "x"],
                c="blue", alpha=0.7, label="Normal")
    plt.scatter( df.loc[df["is_outlier"], "y"], df.loc[df["is_outlier"], "x"], 
                c="red", alpha=0.7, label="Outlier")

    plt.ylabel("Isopos"+ str(i))
    plt.xlabel("Años")
    plt.title("Outliers destacados")
    plt.legend()
    plt.show()
    return

for i in columnas:
    plot_detect_outliers_IQR_year(df_nuevo_4,i,df_fechas_normal,'original_data',detect_outliers_IQR)


