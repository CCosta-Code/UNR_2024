
## Leer parquet file
url = "data_NMV_GAds.parquet"
df_parquet = pd.read_parquet(url)

print(f'Rows: {df_raw.shape[0]}\nColumns: {df_raw.shape[1]}')
print(df_raw.head())
print(df_raw.info())
print(df_raw.columns)
print('Frecuencia según Cyber days: ', df_raw['is_cyber'].value_counts())
print('Número de registros por campaña: ', df_raw['campaign'].value_counts())
print('Min and max dates for campaign: ', df_raw.groupby('campaign')['date'].agg(['min','max','count']))

# Crear variable de fin de semana/día de semana
df_raw['is_week_day'] = df_raw['date'].apply(lambda x: int(x.dayofweek < 5))
print(df_raw[['date','is_week_day']])

print(df_raw[(df_raw.is_cyber==1)].groupby('campaign')['budget_USD'].agg(['min','max','count','sum']))

# Datos perdidos
from dataprep.eda import plot_missing
plot_missing(df_raw)

camp_faltantes = conteo_registros[conteo_registros == 582].index
df_faltantes = df_raw[df_raw['campaign'].isin(camp_faltantes)]

# Crear una ventana de fecha de la fecha mínimo al máximo para cada campaña
fechas_rango = df_faltantes.groupby('campaign')['date'].agg([min, max])

# Iterar sobre campañas con MD
for camp in camp_faltantes:
    rango_fechas = pd.date_range(start=fechas_rango.loc[camp, 'min'], end=fechas_rango.loc[camp, 'max'])
    fechas_faltantes = rango_fechas.difference(df_faltantes[df_faltantes['campaign'] == camp]['date'])

    if not fechas_faltantes.empty:
        print(f"Para la campaña {camp}, la fecha perdida es: {fechas_faltantes}")


# Duplicados
df_raw_unique = df_raw.drop_duplicates(keep='first')

# Métricas simples para la venta y el presupuesto por campaña
print(df_raw.groupby('campaign')[['nmv_USD', 'budget_USD']].agg(['count','min', 'max', 'sum','mean']))


# Estadísticas para el conjunto de datos (similar a describe)
df_raw = df_raw.set_index('date')
df_raw.index = pd.to_datetime(df_raw.index)
from skimpy import skim
skim(df_raw[['is_event','is_cyber','clicks','impressions','visitors','visitors_pur','visits','orders','total_units_net','budget_USD','nmv_USD','cost_USD']])


#### ----- Gráficos ----- ####

## Gráficos Box-plot por variable ##
# Variables a graficar
variables = ['budget_USD','cost_USD','nmv_USD','clicks','impressions','visitors','visitors_pur','visits','orders','total_units_net']

# Estilo
sns.set_style('whitegrid')
fig, axs = plt.subplots(5, 2, figsize=(20, 20))

# Iterar sobre cada variable y graficar el box plot en el correspondiente sub-gráfico
for i, var in enumerate(variables):
    j = i%(5)
    k = i%(2)
    sns.boxplot(x=df_raw[var], ax=axs[j,k], color='powderblue',
                boxprops=dict(edgecolor='royalblue'),
                whiskerprops=dict(color='royalblue'),
                capprops=dict(color='royalblue'),
                medianprops=dict(color='royalblue'),
                flierprops=dict(markerfacecolor='lightgray'))

    axs[j,k].set_xlabel('Valores')
    axs[j,k].set_title(var)

# Ajustar el layout para prevenir overlapping
plt.tight_layout()

# Mostrar el gráfico
plt.show()


##  Histogramas por variable ##
variables = ['clicks','impressions','visitors','visitors_pur','visits','orders','total_units_net','budget_USD','cost_USD','nmv_USD']

sns.set_style('whitegrid')

min_max_values = df_raw[variables].agg(['min', 'max'])

for var in variables:
    plt.figure(figsize=(12, 3))
    # Graficar el histograma
    plt.hist(df_raw[var], bins=50, alpha=0.7, color='powderblue', density=True, label=f'Histograma de {var}')    #'mediumaquamarine'

    # Gráficar la línea KDE (Kernel Density Estimation) 
    sns.kdeplot(df_raw[var], color='royalblue', label=f'KDE de {var}')  #'teal'

    # Registrar los valores min and max sobre los puntos
    plt.axvline(min_max_values.loc['min', var], color='gray', linestyle='--')
    plt.text(min_max_values.loc['min', var], plt.ylim()[1]*1, f'Min: {min_max_values.loc["min", var]:,.0f}', color='gray')

    plt.axvline(min_max_values.loc['max', var], color='gray', linestyle='--')
    plt.text(min_max_values.loc['max', var], plt.ylim()[1]*1, f'Max: {min_max_values.loc["max", var]:,.0f}', color='gray')

    plt.xlabel('Valores', labelpad=20, loc='left')
    plt.ylabel('Frecuencia', labelpad=20, loc='top')
    plt.title(var)
    plt.grid(False)

    sns.despine(top=True, right=True, left=False, bottom=False)  # Delete lines

    plt.show()



## Distribución en el tiempo de la venta neta (NMV) para cada campaña ##
x_='nmv_USD'

cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(25,20))

for i, campaign in enumerate([*df_raw.campaign.unique()]):
    j = i%(rows)
    k = i%(cols)

    df = df_raw[(df_raw['campaign']==campaign)]

    axs[j,k].set_title(campaign, fontsize=16, loc='left')
    if (j==0)&(k==0):
        axs[j,k].plot(df.index, np.log10(1.0+df[x_]), label= f'{x_}', color='royalblue')
        axs[j,k].legend()
    else:
        axs[j,k].plot(df.index, np.log10(1.0+df[x_]), color='royalblue')

    axs[j, k].set_xlabel('Fecha', labelpad=20, loc='left')
    axs[j, k].set_ylabel(f'{x_}', labelpad=20)
    axs[j, k].yaxis.set_label_coords(-0.1, 0.9)  
    axs[j, k].grid(False)

plt.tight_layout()
plt.show()



## Box-plots de la venta neta según día de la semana ##
import calendar
day_names = list(calendar.day_name)

x_ = 'nmv_USD'

campaigns = df_raw['campaign'].unique()
cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(20, 15))   

for i, campaign in enumerate(campaigns):
    j = i // cols
    k = i % cols

    df = df_raw[df_raw['campaign'] == campaign]

    palette = sns.color_palette("viridis", n_colors=len(df['week_day'].unique()))

    axs[j, k].set_title(campaign, fontsize=14, loc='left')
    sns.boxplot(data=df, x='week_day', y=x_, palette=palette, ax=axs[j, k])
    axs[j, k].set_xticklabels(day_names)
    axs[j, k].set_ylabel(f'{x_}', labelpad=20)
    axs[j, k].set_xlabel('Día de la semana', labelpad=20, loc='left')
    axs[j, k].yaxis.set_label_coords(-0.15, 0.9)  
    axs[j, k].grid(False)

plt.tight_layout()
sns.despine(top=True, right=True, left=False, bottom=False)

plt.show()


## Box-plots de la venta neta día del mes ##
x_ = 'nmv_USD'

campaigns = df_raw['campaign'].unique()
cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(35, 30))

for i, campaign in enumerate(campaigns):
    j = i // cols
    k = i % cols

    df = df_raw[df_raw['campaign'] == campaign]

    palette = sns.color_palette("viridis", n_colors=len(df['month_day'].unique()))

    axs[j, k].set_title(campaign, fontsize=16, loc='left')
    sns.boxplot(data=df, x='month_day', y=x_, palette=palette, ax=axs[j, k])
    axs[j, k].set_ylabel(f'{x_}', labelpad=20)
    axs[j, k].set_xlabel('Día del mes', labelpad=20, loc='left')
    axs[j, k].yaxis.set_label_coords(-0.15, 0.9)  
    axs[j, k].grid(False)
plt.tight_layout()

plt.show()


## Ventas netas en el tiempo por campaña según si es un día regular, de evento o cyber ##
y_ = 'nmv_USD'

cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(25,20))
sns.set_style('whitegrid')

for i, campaign in enumerate([*df_raw.campaign.unique()]):
    j = i%(rows)
    k = i%(cols)

    df = df_raw[(df_raw['campaign']==campaign)]

    dfev=df[(df.is_event==1)&(df.is_cyber==0)]
    dfcy=df[df.is_cyber==1]
    dfne=df[df.is_event==0]

    axs[j,k].set_title(campaign, fontsize=14, loc='left')
    if (j==0)&(k==0):
        axs[j,k].scatter(dfev.index, np.log10(1.0 + dfev[y_]), s=11, color='royalblue', edgecolors='royalblue', label='Evento')
        axs[j,k].scatter(dfcy.index, np.log10(1.0 + dfcy[y_]), s=11, color='orangered', edgecolors='orangered', label='Cyber')
        axs[j,k].scatter(dfne.index, np.log10(1.0 + dfne[y_]), s=10, color='lightgray', edgecolors='lightgray', label='Regular') 
        axs[j,k].legend()
    else:
        if (j==0)&(k==1):
            axs[j,k].scatter(dfev.index, np.log10(1.0 + dfev[y_]), s=11, color='royalblue',edgecolors='royalblue')
            axs[j,k].scatter(dfcy.index, np.log10(1.0 + dfcy[y_]), s=11, color='orangered',edgecolors='orangered')
            axs[j,k].scatter(dfne.index, np.log10(1.0 + dfne[y_]), s=10, color='lightgray',edgecolors='lightgray')
        else:
            axs[j,k].scatter(dfev.index, np.log10(1.0 + dfev[y_]), s=11, color='royalblue',edgecolors='royalblue')
            axs[j,k].scatter(dfcy.index, np.log10(1.0 + dfcy[y_]), s=11, color='orangered',edgecolors='orangered')
            axs[j,k].scatter(dfne.index, np.log10(1.0 + dfne[y_]), s=10, color='lightgray',edgecolors='lightgray')
    axs[j,k].set_xlabel('Fecha', labelpad=20, loc='left')
    axs[j,k].set_ylabel(f'{y_}', labelpad=20, loc='top')
    axs[j,k].grid(False)
plt.tight_layout()


## Distribución de las ventas netas por campaña según si es un día regular, de evento o cyber ##
cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(25,15))
sns.set_style('whitegrid')

for i, campaign in enumerate([*df_raw.campaign.unique()]):
    j = i%(rows)
    k = i%(cols)
    df = df_raw[(df_raw['campaign']==campaign)]

    dfev=df[(df.is_event==1)&(df.is_cyber==0)]
    dfcy=df[df.is_cyber==1]
    dfne=df[df.is_event==0]

    axs[j,k].set_title(campaign, fontsize=14, loc='left')
    if (j==0)&(k==0):
        v = dfev[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='royalblue', linewidth=2, alpha=0.8, label='Evento')
        axs[j,k].axvline(v.mean(), color='royalblue', linestyle='--', linewidth=2, label='Media')
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.2, color='royalblue', label='Std.')

        v = dfcy[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='orangered', linewidth=2, alpha=0.8, label='Cyber')
        axs[j,k].axvline(v.mean(), color='orangered', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.1, color='orangered')

        v = dfne[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='silver', linewidth=2, alpha=0.9, label='Regular')
        axs[j,k].axvline(v.mean(), color='silver', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.3, color='silver')
        axs[j, k].legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=0.5)
    else:
        v = dfev[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='royalblue', linewidth=2, alpha=0.8)
        axs[j,k].axvline(v.mean(), color='royalblue', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.2, color='royalblue')

        v = dfcy[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='orangered', linewidth=2, alpha=0.8)
        axs[j,k].axvline(v.mean(), color='orangered', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.1, color='orangered')

        v = dfne[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='silver', linewidth=2, alpha=0.9)
        axs[j,k].axvline(v.mean(), color='silver', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.3, color='silver')

    axs[j,k].set_xlabel(f'{x_}', labelpad=20, loc='left')
    axs[j,k].set_ylabel('Densidad', labelpad=20, loc='top')
    axs[j,k].grid(False)

plt.tight_layout()


## Ventas netas por campaña según si es un día de semana o de fin de semana ##
y_ = 'nmv_USD'

cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(25,20))
sns.set_style('whitegrid')

for i, campaign in enumerate([*df_raw.campaign.unique()]):
    j = i%(rows)
    k = i%(cols)

    df = df_raw[(df_raw['campaign']==campaign)]

    dfnw=df[(df.is_week_day==1)]   
    dfw=df[df.is_week_day==0 & (df.is_cyber==0)] # 0 -> weekend

    axs[j,k].set_title(campaign, fontsize=14, loc='left')
    if (j==0)&(k==0):
        axs[j,k].scatter(dfnw.index, np.log10(1.0 + dfnw[y_]), s=10, color='lightgray', edgecolor='lightgray', label='Semana')
        axs[j,k].scatter(dfw.index, np.log10(1.0 + dfw[y_]), s=11, color='royalblue', edgecolor='royalblue', label='Fin de semana')
        axs[j,k].legend()
    else:
        if (j==0)&(k==1):
            axs[j,k].scatter(dfnw.index, np.log10(1.0 + dfnw[y_]), s=10, color='lightgray', edgecolor='lightgray')
            axs[j,k].scatter(dfw.index, np.log10(1.0 + dfw[y_]), s=11, color='royalblue', edgecolor='royalblue')
        else:
            axs[j,k].scatter(dfnw.index, np.log10(1.0 + dfnw[y_]), s=10, color='lightgray', edgecolor='lightgray') #'skyblue'
            axs[j,k].scatter(dfw.index, np.log10(1.0 + dfw[y_]), s=11, color='royalblue', edgecolor='royalblue')

    axs[j,k].set_xlabel('Fecha', labelpad=20, loc='left')
    axs[j,k].set_ylabel(f'{y_}', labelpad=20, loc='top')
    axs[j,k].grid(False)

plt.tight_layout()


## Distribución de las ventas netas por campaña según día se semana o fin de semana ##
cols = 3
rows = 4
fig, axs = plt.subplots(rows, cols, figsize=(25,15))
fig.patch.set_facecolor('white')

for i, category in enumerate([*df_raw.campaign.unique()]):
    j = i%(rows)
    k = i%(cols)

    df = df_raw[(df_raw['campaign']==category)]
    dfwd=df[(df.is_week_day==1)&(df.is_cyber==0)]  # week day
    dfw=df[(df.is_week_day==0)&(df.is_cyber==0)]   # weekend

    axs[j,k].set_title(category, fontsize=14, loc='left')
    if (j==0)&(k==0):
        v = dfwd[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='silver', linewidth=2, alpha=0.9, label='Semana')
        axs[j,k].axvline(v.mean(), color='silver', linestyle='--', linewidth=2, label='Media')
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.3, color='silver', label='Std.')

        v = dfw[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='royalblue', linewidth=2, alpha=0.9, label='Fin de semana')
        axs[j,k].axvline(v.mean(), color='royalblue', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.2, color='royalblue')
        axs[j,k].legend()
    else:
        v = dfwd[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='silver', linewidth=2, alpha=0.9)
        axs[j,k].axvline(v.mean(), color='silver', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.3, color='silver') #'skyblue'
        v = dfw[x_]
        kde = stats.gaussian_kde(v.values)
        xx = np.linspace(0, max(v), 1000)
        _ = axs[j,k].plot(xx, kde(xx), color='royalblue', linewidth=2, alpha=0.9)
        axs[j,k].axvline(v.mean(), color='royalblue', linestyle='--', linewidth=2)
        axs[j,k].axvspan(v.mean()-v.std()/2, v.mean()+v.std()/2, alpha=0.2, color='royalblue')

    axs[j,k].set_xlabel(f'{x_}', labelpad=20, loc='left')
    axs[j,k].set_ylabel('Densidad', labelpad=20, loc='top')
    axs[j,k].grid(False)

plt.tight_layout()


#### ----- KPIs de Marketing ----- ####

## CPC
df_raw['quarter'] = df_raw.index.to_period('Q')
# Convertir el índice a período trimestral - quarter
df_raw['CPC'] = df_raw['cost_USD'] / df_raw['clicks']
df_raw['CPC'] = df_raw['CPC'].replace([np.inf, -np.inf], np.nan).fillna(0)

# CPC promedio por campaña para días regulares
# Agrupar por campaña y trimestre, y calcular el CPC promedio por campaña y trimestre
cpc_per_campaign_quarter = df_raw[df_raw.is_cyber == 0].groupby(['campaign', 'quarter']).apply(lambda x: (x['cost_USD'].sum() / x['clicks'].sum())).reset_index()
cpc_per_campaign_quarter.columns = ['campaign', 'quarter', 'CPC']
# CPC promedio por campaña para días cyber
cpc_per_campaign_quarter2 = df_raw[df_raw.is_cyber == 1].groupby(['campaign', 'quarter']).apply(lambda x: (x['cost_USD'].sum() / x['clicks'].sum())).reset_index()
cpc_per_campaign_quarter2.columns = ['campaign', 'quarter', 'CPC']

# Comparación por trimestre del CPC promedio por campaña según quarter
campaigns = cpc_per_campaign_quarter['campaign'].unique()

for campaign in campaigns:
    subset = cpc_per_campaign_quarter[cpc_per_campaign_quarter['campaign'] == campaign]
    subset2 = cpc_per_campaign_quarter2[cpc_per_campaign_quarter2['campaign'] == campaign]

    plt.figure(figsize=(12, 3))
    sns.set(style="whitegrid")

    # Graficar la serie 'Regular'
    plt.plot(subset['quarter'].astype(str), subset['CPC'], marker='o', linewidth=2, color='silver', label='No cyber')
    plt.annotate(f"{subset['CPC'].iloc[0]:.2f}", xy=(subset['quarter'].iloc[0].strftime('%YQ%q'), subset['CPC'].iloc[0]),
                 xytext=(5, 5), textcoords='offset points', color='silver', fontsize=9)
    plt.annotate(f"{subset['CPC'].iloc[-1]:.2f}", xy=(subset['quarter'].iloc[-1].strftime('%YQ%q'), subset['CPC'].iloc[-1]),
                 xytext=(5, 5), textcoords='offset points', color='silver', fontsize=9)

    # Graficar la serie 'Cyber'
    plt.plot(subset2['quarter'].astype(str), subset2['CPC'], marker='o', linewidth=2, color='orangered', label='Cyber')
    plt.annotate(f"{subset2['CPC'].iloc[0]:.2f}", xy=(subset2['quarter'].iloc[0].strftime('%YQ%q'), subset2['CPC'].iloc[0]),
                 xytext=(5, 5), textcoords='offset points', color='orangered', fontsize=9)
    plt.annotate(f"{subset2['CPC'].iloc[-1]:.2f}", xy=(subset2['quarter'].iloc[-1].strftime('%YQ%q'), subset2['CPC'].iloc[-1]),
                 xytext=(5, 5), textcoords='offset points', color='orangered', fontsize=9)

    plt.xlabel('Trimestre', labelpad=20, loc='left')
    plt.ylabel('CPC (USD)', labelpad=20, loc='top')
    plt.title(f'{campaign}', fontsize=14, loc='left')
    plt.xticks()
    plt.legend()
    plt.tight_layout()
    plt.grid(False)

    sns.despine(top=True, right=True, left=False, bottom=False)  # Delete line

    # Mostrar el gráfico
    plt.show()

## CTR
# CTR en día regulares por campaña y quarter
df_ctr_quarter = df_raw[df_raw.is_cyber == 0].groupby(['campaign', 'quarter']).agg({
    'clicks': 'sum',        # Total de clics en la campaña
    'impressions': 'sum'    # Total de impresiones (visualizaciones) en la campaña
}).reset_index()

df_ctr_quarter['CTR'] = (df_ctr_quarter['clicks'] / df_ctr_quarter['impressions']) * 100
print(df_ctr_quarter[['campaign', 'quarter', 'CTR']].sort_values(by=['campaign', 'quarter', 'CTR'], ascending=[True, True, False]))

# CTR en día cyber por campaña y quarter
df_ctr2 = df_raw[df_raw.is_cyber==1].groupby('campaign').agg({
    'clicks': 'sum',        # Total de clics en la campaña
    'impressions': 'sum'    # Total de impresiones (visualizaciones) en la campaña
})

df_ctr_quarter2['CTR'] = (df_ctr_quarter2['clicks'] / df_ctr_quarter2['impressions']) * 100
print(df_ctr_quarter2[['campaign', 'quarter', 'CTR']].sort_values(by=['campaign', 'quarter', 'CTR'], ascending=[True, True, False]))

# Gráfico
campaigns = df_ctr_quarter2['campaign'].unique()

for campaign in campaigns:
    subset = df_ctr_quarter[df_ctr_quarter['campaign'] == campaign]
    subset2 = df_ctr_quarter2[df_ctr_quarter2['campaign'] == campaign]

    plt.figure(figsize=(12, 3))
    sns.set(style="whitegrid")

    plt.plot(subset['quarter'].astype(str), subset['CTR'], marker='o', linewidth=2, color='silver', label='No cyber')
    # Agregar etiquetas al primer y último valor de 'Regular'
    plt.annotate(f"{subset['CTR'].iloc[0]:.2f}", xy=(subset['quarter'].iloc[0].strftime('%YQ%q'), subset['CTR'].iloc[0]),
                 xytext=(5, 5), textcoords='offset points', color='silver', fontsize=9)
    plt.annotate(f"{subset['CTR'].iloc[-1]:.2f}", xy=(subset['quarter'].iloc[-1].strftime('%YQ%q'), subset['CTR'].iloc[-1]),
                 xytext=(5, 5), textcoords='offset points', color='silver', fontsize=9)

    plt.plot(subset2['quarter'].astype(str), subset2['CTR'], marker='o', linewidth=2, color='orangered', label='Cyber')
    # Agregar etiquetas al primer y último valor de 'Cyber'
    plt.annotate(f"{subset2['CTR'].iloc[0]:.2f}", xy=(subset2['quarter'].iloc[0].strftime('%YQ%q'), subset2['CTR'].iloc[0]),
                 xytext=(5, 5), textcoords='offset points', color='orangered', fontsize=9)
    plt.annotate(f"{subset2['CTR'].iloc[-1]:.2f}", xy=(subset2['quarter'].iloc[-1].strftime('%YQ%q'), subset2['CTR'].iloc[-1]),
                 xytext=(5, 5), textcoords='offset points', color='orangered', fontsize=9)

    plt.xlabel('Trimestre', labelpad=20, loc='left')
    plt.ylabel('CTR (%)', labelpad=20, loc='top')
    plt.title(f'{campaign}', fontsize=14, loc='left')
    plt.xticks()
    plt.legend()
    plt.tight_layout()
    plt.grid(False)

    sns.despine(top=True, right=True, left=False, bottom=False)  #
    plt.show()


## Convertion Rate
# CR en día regulares por campaña y quarter
df_conversion_quarter = df_raw[df_raw.is_cyber == 0].groupby(['campaign', 'quarter']).agg({
    'orders': 'sum',      # Total de órdenes realizadas
    'visitors': 'sum'     # Total de visitantes
}).reset_index()

# Calcular Conversion Rate
df_conversion_quarter['conversion_rate'] = (df_conversion_quarter['orders'] / df_conversion_quarter['visitors']) * 100
print(df_conversion_quarter[['campaign', 'quarter', 'conversion_rate']].sort_values(by=['campaign', 'quarter', 'conversion_rate'], ascending=[True, True, False]))

# CR en día cyber por campaña y quarter
df_conversion_quarter2 = df_raw[df_raw.is_cyber == 1].groupby(['campaign', 'quarter']).agg({
    'orders': 'sum',      # Total de órdenes realizadas
    'visitors': 'sum'     # Total de visitantes
}).reset_index()

df_conversion_quarter2['conversion_rate'] = (df_conversion_quarter2['orders'] / df_conversion_quarter2['visitors']) * 100
print(df_conversion_quarter2[['campaign', 'quarter', 'conversion_rate']].sort_values(by=['campaign', 'quarter', 'conversion_rate'], ascending=[True, True, False]))


## ROI % por campaña
# ROI en día regular por campaña y quarter
df_roi_quarter = df_raw[df_raw.is_cyber == 0].groupby(['campaign', 'quarter']).agg({
    'nmv_USD': 'sum',     # Ingresos generados por la campaña
    'cost_USD': 'sum'     # Costos totales de la campaña
}).reset_index()

# Calcular el ROI
df_roi_quarter['ROI'] = (df_roi_quarter['nmv_USD'] - df_roi_quarter['cost_USD']) / df_roi_quarter['cost_USD'] *100
print(df_roi_quarter[['campaign', 'quarter', 'ROI']].sort_values(by=['campaign', 'quarter', 'ROI'], ascending=[True, True, False]))

# ROI en día cyber por campaña y quarter
df_roi_quarter2 = df_raw[df_raw.is_cyber == 1].groupby(['campaign', 'quarter']).agg({
    'nmv_USD': 'sum',     # Ingresos generados por la campaña
    'cost_USD': 'sum'     # Costos totales de la campaña
}).reset_index()

df_roi_quarter2['ROI'] = (df_roi_quarter2['nmv_USD'] - df_roi_quarter2['cost_USD']) / df_roi_quarter2['cost_USD'] *100
print(df_roi_quarter2[['campaign', 'quarter', 'ROI']].sort_values(by=['campaign', 'quarter', 'ROI'], ascending=[True, True, False]))


"""Para realizar el análisis comparando las métricas para días de semana vs fines de semana simplemente cambiar el filtro a:
    df_raw.is_week_day == 1 para día de semana
    df_raw.is_week_day == 1 para fin de semana
"""