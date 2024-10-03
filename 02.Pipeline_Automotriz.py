# Leer parquet file
url = "/home/jupyter/UNR/data/data_NMV_GAds_manipulated.parquet"

df_parquet = pd.read_parquet(url)

df_raw = df_parquet
print(f'Rows: {df_raw.shape[0]}\nColumns: {df_raw.shape[1]}')


#### ----- Identificación de outliers e imputación ----- ####
variables = ['nmv_USD','clicks','visitors','visitors_pur','visits','orders','total_units_net','impressions','budget_USD','cost_USD']
cont_param = 0.20
df_raw_ = df_raw.copy()
df_raw_.reset_index(drop=False, inplace=True)
df_full = df_raw_[df_raw_['campaign'] == 'Automotriz']
ind_completo = df_full.index
# Inicializar un DataFrame para almacenar los resultados con el mismo índice que df_full
outliers_data = pd.DataFrame(index=df_full.index)
# Lista para almacenar los resultados de cada variable
results = []

for var in variables:

    decomposed = sm.tsa.seasonal_decompose(df_full[var], model='multiplicative', period=30)
    outliers_data[var + '_residual'] = decomposed.resid
    outliers_data[var + '_trend'] = decomposed.trend
    outliers_data[var + '_seasonal'] = decomposed.seasonal
    stl = outliers_data.copy()
    stl[var + '_tr_seas'] = stl[var + '_trend'] * stl[var + '_seasonal'] # Trend * Seasonal components
    # Agrego el campo de cyber --> Excluyo días de cyber --> Busco outliers para días sin cyber
    outliers_data = df_full[['is_cyber']].merge(outliers_data, left_index=True, right_index=True, how='left')
    # df completo (cyber 0 y 1) - Guardar una copia
    outliers_data_cyber = outliers_data.copy()
    # df cyber = 0 y elimino el campo
    outliers_data = outliers_data[outliers_data.is_cyber == 0]
    outliers_data = outliers_data.drop('is_cyber', axis = 1)
    outliers_data = outliers_data.dropna()

    # Inicializar el modelo COPOD
    model = COPOD(contamination=cont_param)   #  meaning it expects 20% of the data to be outliers
    model.fit(outliers_data[[var + '_residual']])
    outliers = model.predict(outliers_data[[var + '_residual']])
    outliers_data[var + '_outlier'] = outliers

    # Realizar interpolación para los registros considerados outliers
    # Columna con residuos a imputar
    imputed_column = var + '_residual'
    # Selecciono el conjunto sin outliers
    df_no_out = outliers_data[outliers_data[var + '_outlier'] == 0]
    df_no_out = df_no_out.reindex(ind_completo)
    impute_values = df_no_out[imputed_column].interpolate(method='quadratic', limit_direction='both')

    # Agrego la info de is_cyber
    # Al df que tiene la descomposición de los cyber tbn le pego "_residual"
    outliers_data_full = df_full[['is_cyber','date','campaign']].merge(outliers_data, left_index=True, right_index=True, how='left')
    # Crear un flag para identificar casos tanto de outliers como de cyber days a ser imputados
    outliers_data_full[var + '_flag'] = np.where((outliers_data_full['is_cyber'] == 1) | (outliers_data_full[var + '_outlier'] == 1), 1, 0)
    outliers_indices = outliers_data_full[outliers_data_full[var + '_flag'] == 1].index

    # Nombre de la columna con datos imputados
    imputed_column_name = imputed_column.replace('_residual', '_imputed')
    # Busca los índices donde hay outliers y cyber days y asignar los valores imputados
    outliers_data_full.loc[outliers_indices, imputed_column_name] = impute_values[outliers_indices]
    outliers_data_full[var + '_imputed'] = outliers_data_full[var + '_imputed'].abs()
    # Cuando hay un outlier que asigne a la columna de residuos nuevos el valor imputado, c.c. que asigne el residuo original
    outliers_data_full[var + '_residual2'] = outliers_data_full.apply(lambda row: row[var + '_imputed']
                                   if ((row[var + '_flag'] == 1) & (row[var + '_imputed'] >= 0)) else row[var + '_residual'], axis=1)
    outliers_data_stl = stl[[var + '_tr_seas']].merge(outliers_data_full, left_index=True, right_index=True, how='right')
    # Nueva variable final que reemplaza a la original
    outliers_data_stl[var + '_2'] = outliers_data_stl.apply(lambda row: row[var + '_residual2'] * row[var + '_tr_seas'], axis=1)
    outliers_data_stl[var + '_2']  = round(outliers_data_stl[var + '_2'], 0)

    # Eliminar las columnas adicionales generadas
    columnas_a_eliminar = [var + '_tr_seas', var + '_trend', var + '_residual', var + '_residual2', var +  '_seasonal', var +  '_imputed', \
                           var + '_outlier', 'is_cyber']
    outliers_data_stl.drop(columnas_a_eliminar, axis=1, inplace=True)

    # Agregar resultados a la lista
    results.append(outliers_data_stl)

df_imp = pd.concat(results, ignore_index=True)

for var in variables:
    df_imp[var + '_imp'] = df_imp.apply(lambda row: row[var + '_2']
                                   if ((row[var + '_flag'] == 1)) else row[var], axis=1)

# Eliminar eventos y variables originales (sin tratamiento de outliers e imp)
column = ['is_event','is_cyber', 'previous_cyber_days', 'clicks', 'impressions', 'visitors',
       'visitors_pur', 'visits', 'orders', 'total_units_net',
       'budget_USD', 'cost_USD', 'nmv_USD', 'clicks_flag', 'budget_USD_flag','cost_USD_flag', 'impressions_flag', 
       'nmv_USD_flag', 'orders_flag', 'total_units_net_flag', 'visitors_flag', 'visitors_pur_flag','visits_flag']

data = df_imp.drop(column, axis=1)

# Eliminar las columnas con "_2"
for column in df_imp.columns:
    if column.endswith('_2'):
        data.drop(column, axis=1, inplace=True)
# Eliminar las columnas del OHE para campañas
for column in df_imp.columns:
    if column.startswith('campaign_'):
        data.drop(column, axis=1, inplace=True)
print('Columnas finales: ',data.columns)



#### ----- Separación Train/Test ----- ####
def train_test_per_campaigns(df, train_frac):
    '''
    Crete the train-test datasets for each campaign.
        Parameters:
            df (DataFrame): Table with campaign's historical data.
            train_frac (float): Test fraction for the split.
        Returns:
            df (DataFrame): Table with campaign's historical data labeled
                with the "dataset" column.
    '''
    # Flag train-test
    d_list = []
    for campaign in df['campaign'].unique():
        df_tmp = df[(df['campaign'] == campaign)]
        df_tmp.reset_index(drop=True, inplace=True)
        itrain = int(train_frac * df_tmp.shape[0])
        df_train, df_test =  df_tmp[:itrain].copy(), df_tmp[itrain+3:].copy()    
        df_train['dataset'], df_test['dataset'] = 'train', 'test'
        d_list.append(pd.concat([df_train, df_test]))
    df = pd.concat(d_list)
    return df.reset_index(drop=True)

# Correr función
train_frac = 0.8   
df = train_test_per_campaigns(data, train_frac)
print('Número de registros en cada data set', df['dataset'].value_counts())

## Gráfico de separación train y test para cada variable
# Variables to plot
cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(25,20))
sns.set_style('whitegrid')

for i, var in enumerate(variables):
    j = i%(rows)
    k = i%(cols)

    df_ = df.copy()
    df_['date'] = pd.to_datetime(df_['date'])
    df1=df_[df_.dataset == 'train'].set_index('date')
    df2=df_[df_.dataset == 'test'].set_index('date')

    axs[j,k].set_title(var, fontsize=14, loc='left')
    if (j==0)&(k==0):

        # Ploteando los conjuntos de entrenamiento y prueba
        axs[j,k].plot(df1.index, df1[var], label='Training', color='silver')
        axs[j,k].plot(df2.index, df2[var], label='Test', color='royalblue')

        # Línea vertical para marcar el punto de división entre train y test
        axs[j,k].axvline(pd.to_datetime('2023-05-01'), color='darkgray', ls='--')
        axs[j,k].legend(['Training Set', 'Test Set'])
    else:
        if (j==0)&(k==1):
            axs[j,k].plot(df1.index, df1[var], label='Training', color='silver')
            axs[j,k].plot(df2.index, df2[var], label='Test', color='royalblue')
            axs[j,k].axvline(pd.to_datetime('2023-05-01'), color='darkgray', ls='--')
        else:
            axs[j,k].plot(df1.index, df1[var], label='Training', color='silver')
            axs[j,k].plot(df2.index, df2[var], label='Test', color='royalblue')
            axs[j,k].axvline(pd.to_datetime('2023-05-01'), color='darkgray', ls='--')

    axs[j,k].set_xlabel('Fecha', labelpad=20, loc='left')
    axs[j,k].set_ylabel(f'{var}', labelpad=20, loc='top')
    axs[j,k].grid(False)
plt.tight_layout()



#### ----- Correlaciones ----- ####
def corr_features(data, corr_threshold):
    '''
    Drop highly correlated features.
        Parameters:
            data (DataFrame): Table with GAds history per campaign.
        Returns:
            _ (list): List of features to drop from the table.
    '''
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)
    return [c for c in tri_df.columns if any(tri_df[c] >  corr_threshold)]

# Parámetros
target = ['nmv_UDS_imp']
response_features = [] 
# Feature selection
corr_threshold = 0.98 

# Seleccionar variables
features = [c for c in df.columns if c not in ['date','campaign','week_day', 'month_day', 'year_day', 'month_year','is_week_day'] + response_features]
# Calcular la matriz de correlación
corr_matrix = round(data[features].corr(), 2)
styled_corr = corr_matrix.style.background_gradient(cmap='Blues') 
styled_corr = styled_corr.set_caption("<span style='font-size: 16px'>Matriz de correlaciones</span>")
styled_corr = styled_corr.set_properties(**{'font-size': '12px'})
# Mostrar la matriz de correlación
print(styled_corr)

# Aplicar función para eliminar variables dependiendo de la correlación
to_drop = corr_features(df[features], corr_threshold)
data = df.drop(to_drop, axis=1)
print('Cantidad de variables a eliminar: ', len(to_drop))
print('Variables a eliminar: ',to_drop)



#### ----- Lags y rolling means ----- ####
campaigns = data.campaign.unique()
window_sizes = [7, 14]
lag_days = [1, 2, 3]

variables = ['clicks_imp', 'impressions_imp', 'visits_imp', 'orders_imp', 'total_units_net_imp', 'budget_USD_imp', 'cost_USD_imp', 'nmv_USD_imp']

process_dfs = []

for camp in campaigns:
    df_camp = data[data.campaign == camp].copy()
    df_camp = df_camp.sort_values(by='date')

    for var in variables:
        for w in window_sizes:
            df_camp[f'mean_{w}d_{var}'] = df_camp.groupby('campaign')[var].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
            df_camp[f'mean_{w}d_{var}'] = round(df_camp[f'mean_{w}d_{var}'], 2)
    
        for lag in lag_days:
                df_camp[f'lag_{lag}d_{var}'] = df_camp.groupby('campaign')[var].shift(lag)
                df_camp[f'lag_{lag}d_{var}'] = round(df_camp[f'lag_{lag}d_{var}'], 2)
    

    process_dfs.append(df_camp)
df_means = pd.concat(process_dfs)
df = df_means.copy()



#### ----- Ajuste de modelos ----- ####
## XGBoost
# General
target = ['nmv_USD_imp']
response_features = [] 
param_grid = {
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':[40,50,75,100],
    'max_depth':[2,5,10],
    'subsample':[0.2,0.4,0.6,0.8],

}

## Definir variables predictoras
columns_predictors = [c for c in df.columns if c not in ['date','nmv_USD_imp','campaign'] + response_features]
# El budget tengo que incluirlo sí o sí en el modelo
if 'budget_USD_imp' not in columns_predictors:
    columns_predictors = ['budget_USD_imp'] + columns_predictors
elif columns_predictors[0] != 'budget_USD_imp':
    columns_predictors.remove('budget_USD_imp')
    columns_predictors = ['budget_USD_imp'] + columns_predictors
print('Número de variables finales: ',len(columns_predictors))
print('Variables finales: ', columns_predictors)

campaign = str(df.campaign.unique()[0]) + "_XGBoost"
model_dict = {}
df.sort_values(by='date', inplace = True)

x_train, y_train = df[df['dataset']=='train'][columns_predictors].values, df[df['dataset']=='train'][target].values
x_test, y_test = df[df['dataset']=='test'][columns_predictors].values, df[df['dataset']=='test'][target].values
# Validación de parámetros
fit_params={"verbose":0,
            "eval_set" : [(x_train, y_train),(x_test, y_test)]}
# Iniciar regressor
gbm = xgb.XGBRegressor(early_stopping_rounds = 30, eval_metric = "rmse")
# Iniciar grid searcher
grid = GridSearchCV(estimator = gbm,
                    param_grid = param_grid,
                    scoring = 'neg_mean_squared_error',
                    cv = None)
# Buscar los mejores hyper parameters
results = grid.fit(x_train, y_train, **fit_params)
# Recuperar el mejor modelo
model = results.best_estimator_
# Predicciones en test data
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

model_dict.update({campaign:{
                  'predictors':columns_predictors,
                  'model':grid.best_estimator_,
                  'mape_test':round(100*mean_absolute_percentage_error(y_test, y_pred_test), 2),
                  'mae_test':mean_absolute_error(y_test, y_pred_test),
                  'rmse_test':np.sqrt(mean_squared_error(y_test, y_pred_test)),
                  'mape_train':round(100*mean_absolute_percentage_error(y_train, y_pred_train), 2),
                  'mae_train':mean_absolute_error(y_train, y_pred_train),
                  'rmse_train':np.sqrt(mean_squared_error(y_train, y_pred_train)),
                           }})
y_pred_xgb = y_pred_test


## SARIMAX
data = df.drop(df.filter(like='mean_').columns, axis=1)
data = data.drop(data.filter(like='lag_').columns, axis=1)

columns_predictors_s = [c for c in data.columns if c not in ['date','nmv_USD_imp','campaign','week_day','month_day','year_day','month_year','is_week_day','dataset'] + response_features]
# El budget tengo que incluirlo sí o sí en el modelo
if 'budget_USD_imp' not in columns_predictors_s:
    columns_predictors_s = ['budget_USD_imp'] + columns_predictors_s
elif columns_predictors_s[0] != 'budget_USD_imp':
    columns_predictors_s.remove('budget_USD_imp')
    columns_predictors_s = ['budget_USD_imp'] + columns_predictors_s
print('Número de variables finales: ',len(columns_predictors_s))
print('Variables finales: ', columns_predictors_s)

data.sort_values(by='date', inplace = True)
data.set_index('date', inplace=True)

campaign = str(data.campaign.unique()[0]) + "_SARIMAX"
# Train-test split
x_train = data[data['dataset']=='train'][columns_predictors_s]
y_train = data[data['dataset']=='train'][target]
x_test = data[data['dataset']=='test'][columns_predictors_s]
y_test = data[data['dataset']=='test'][target]
# Probar distintas estacionalidades y seleccionar la que mejor ajuste
seasonalities = [4, 12, 24]  

best_mape = float('inf')
best_model = None
best_seasonality = None

for m in seasonalities:

    SARIMAX_model = pm.auto_arima(y_train['nmv_USD_imp'], exogenous=x_train[columns_predictors_s],
                                  start_p=1, start_q=1,
                                  test='adf',
                                  max_p=3, max_q=3, m=m,
                                  start_P=0, seasonal=True,
                                  d=1, D=1, 
                                  trace=False,
                                  error_action='ignore',  
                                  suppress_warnings=True, 
                                  stepwise=True)
     
    y_pred_test = SARIMAX_model.predict(n_periods=len(y_test), exogenous=x_test[columns_predictors_s])
    y_pred_train = SARIMAX_model.predict(n_periods=len(y_train), exogenous=x_train[columns_predictors_s])
    
    mape_test = round(100*mean_absolute_percentage_error(y_test, y_pred_test), 2)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    mape_train = round(100*mean_absolute_percentage_error(y_train, y_pred_train), 2)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    if mape_test < best_mape:
        best_mape = mape_test
        best_model = SARIMAX_model
        best_seasonality = m
        y_pred_sarimax = y_pred_test

model_dict.update({campaign:{
                  'predictors':columns_predictors_s,
                  'model':best_model,
                  'seasonality':best_seasonality,
                  'mape_test':best_mape,
                  'mae_test':mean_absolute_error(y_test, best_model.predict(n_periods=len(y_test), exogenous=x_test[columns_predictors_s])),
                  'rmse_test':np.sqrt(mean_squared_error(y_test, best_model.predict(n_periods=len(y_test), exogenous=x_test[columns_predictors_s]))),
                  'mape_train':round(100*mean_absolute_percentage_error(y_train, best_model.predict(n_periods=len(y_train), exogenous=x_train[columns_predictors_s])), 2),
                  'mae_train':mean_absolute_error(y_train, best_model.predict(n_periods=len(y_train), exogenous=x_train[columns_predictors_s])),
                  'rmse_train':np.sqrt(mean_squared_error(y_train, best_model.predict(n_periods=len(y_train), exogenous=x_train[columns_predictors_s]))),
                           }})


## VARX
df_varx = df.copy()
df_varx.sort_values(by='date', inplace = True)
df_varx.set_index('date', inplace=True)
columns_predictors_mean = ['nmv_USD_imp','mean_7d_nmv_USD_imp'] 

# Train-test split
x_train = df_varx[df_varx['dataset'] == 'train'][columns_predictors_s] 
y_train = df_varx[df_varx['dataset'] == 'train'][columns_predictors_mean]
x_test = df_varx[df_varx['dataset'] == 'test'][columns_predictors_s]
y_test = df_varx[df_varx['dataset'] == 'test'][columns_predictors_mean]

endog_train = y_train  # Variables endógenas
exog_train = x_train  # Variables exógenas

endog_test = y_test
exog_test = x_test

endog_train = endog_train.fillna(endog_train.mean())
exog_train = exog_train.fillna(exog_train.mean())

endog_test = endog_test.fillna(endog_test.mean())
exog_test = exog_test.fillna(exog_test.mean())

# Ajustar un modelo VARMAX (VARX con variables exógenas)
best_mape = float('inf')
best_model = None
best_lag_order = None

for lag in range(1, 14):  
    model = VARMAX(endog_train, exog=exog_train, order=(lag, 0))
    varx_model = model.fit(disp=False)
    y_pred_test = varx_model.forecast(steps=len(endog_test), exog=exog_test)
    y_pred_train = varx_model.forecast(steps=len(endog_train), exog=exog_train)

    mape_test = round(100 * mean_absolute_percentage_error(endog_test, y_pred_test), 2)
    mae_test = mean_absolute_error(endog_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(endog_test, y_pred_test))

    mape_train = round(100 * mean_absolute_percentage_error(endog_train, y_pred_train), 2)
    mae_train = mean_absolute_error(endog_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(endog_train, y_pred_train))

    if mape_test < best_mape:
        best_mape = mape_test
        best_model = varx_model
        best_lag_order = lag
        y_pred_varx = y_pred_test

# Guardar los resultados del mejor modelo
campaign = str(df.campaign.unique()[0]) + "_VARMAX"
model_dict.update({
    campaign: {
        'predictors': columns_predictors_s,
        'model': best_model,
        'lag_order': best_lag_order,
        'mape_test': best_mape,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'mape_train': mape_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train
    }
})



#### ----- Selección del mejor modelo para la campaña en estudio ----- ####
def compare_models(model_dict):
    # Crear un DataFrame con los resultados de todos los modelos
    results = []
    for campaign, data in model_dict.items():
        results.append({
            'campaign': campaign,
            'mape_test': data['mape_test'],
            'mae_test': data['mae_test'],
            'rmse_test': data['rmse_test'],
            'mape_train': data['mape_train'],
            'mae_train': data['mae_train'],
            'rmse_train': data['rmse_train']
        })
    
    df_results = pd.DataFrame(results)
    
    # Calcular el ranking de cada modelo para cada métrica
    for metric in ['mape_test', 'mae_test', 'rmse_test']:
        df_results[f'{metric}_rank'] = df_results[metric].rank()    
    # Calcular el ranking promedio
    df_results['avg_rank'] = df_results[['mape_test_rank', 'mae_test_rank', 'rmse_test_rank']].mean(axis=1)
    
    best_model = df_results.loc[df_results['avg_rank'].idxmin()]  
    print("Resumen de todos los modelos:")
    print(df_results)
    
    print("\nMejor modelo:")
    print(best_model)
    
    # Obtener las predicciones del mejor modelo
    best_model_name = best_model['campaign']
    if best_model_name == 'Automotriz_XGBoost':
        y_pred = y_pred_xgb
    elif best_model_name == 'Automotriz_SARIMAX':
        y_pred = y_pred_sarimax
    elif best_model_name == 'Automotriz_VARMAX':
        y_pred = y_pred_varx
    else:
        raise ValueError(f"Modelo no reconocido: {best_model_name}")
    
    # Agregar las predicciones al diccionario de resultados
    best_model_details = model_dict[best_model_name]
    best_model_details['y_pred'] = y_pred
    
    return best_model_details

# Uso de la función
best_model_details = compare_models(model_dict)
# Guardar las predicciones del mejor modelo
y_pred = best_model_details['y_pred']



#### ----- Gráficos ----- ####
## Función para graficar las ventas netas históricas reales vs predicciones bajo 
## el mejor modelo para la campaña

def plot_results(df, y_pred, var='nmv_USD_imp'):
    plt.figure(figsize=(10, 3))
    sns.set_style('whitegrid')
    
    # Asegurarse de que la fecha esté en el formato correcto
    df['date'] = pd.to_datetime(df['date'])
    
    # Ordenar el DataFrame por fecha
    df = df.sort_values('date')
    
    # Graficar la serie histórica completa
    plt.plot(df['date'], df[var], label='NMV', color='silver')
    
    # Separar los datos de prueba
    df_test = df[df.dataset == 'test']
    
    # Graficar las predicciones
    plt.plot(df_test['date'], y_pred, label='Predicciones', color='royalblue')
    
    # Línea vertical para marcar el punto de inicio de las predicciones
    plt.axvline(df_test['date'].iloc[0], color='darkgray', ls='--')
    
    # Configuración del gráfico
    
    plt.title(f'Automotriz', fontsize=14, loc='left')
    plt.xlabel('Fecha', fontsize=10, loc='left')
    plt.ylabel('nmv_USD', fontsize=10, loc='top')
    plt.legend(fontsize=8)
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()

# Uso de la función
plot_results(df, y_pred) 


## Función para graficar test data set vs predicciones del mejor modelo para la campaña
def plot_results_test(df, y_pred, var='nmv_USD_imp'):
    plt.figure(figsize=(10, 3))
    sns.set_style('whitegrid')
    
    df['date'] = pd.to_datetime(df['date'])    
    df = df.sort_values('date')
    # Separar los datos de prueba
    df_test = df[df.dataset == 'test']

    # Graficar la serie histórica completa
    plt.plot(df_test['date'], df_test[var], label='NMV', color='silver')    
    # Graficar las predicciones
    plt.plot(df_test['date'], y_pred, label='Predicciones', color='royalblue')
  
    plt.title(f'Automotriz', fontsize=14, loc='left')
    plt.xlabel('Fecha', fontsize=10, loc='left')
    plt.ylabel('nmv_USD', fontsize=10, loc='top')
    plt.legend(fontsize=8)
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()

# Uso de la función
plot_results_test(df, y_pred)


## Función para graficar test data set vs predicciones de los 3 modelos para la campaña 
def plot_results_comp(df, y_pred, var='nmv_USD_imp'):
    plt.figure(figsize=(10, 3))
    sns.set_style('whitegrid')
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df_test = df[df.dataset == 'test']

    # Graficar la serie histórica completa
    plt.plot(df_test['date'], df_test[var], label='NMV', color='silver')
    
    # Graficar las predicciones
    plt.plot(df_test['date'], y_pred_xgb, label='XGBoost', color='royalblue')
    plt.plot(df_test['date'], y_pred_sarimax, label='SARIMAX', color='cadetblue',ls='--')
    plt.plot(df_test['date'], y_pred_varx[var], label='VARX', color='cadetblue',ls='dotted')
    
    plt.title(f'Menaje', fontsize=14, loc='left')
    plt.xlabel('Fecha', fontsize=10, loc='left')
    plt.ylabel('nmv_USD', fontsize=10, loc='top')
    plt.legend(fontsize=8)
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()

# Uso de la función
plot_results_comp(df, y_pred)