import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Load datasets
@st.cache
def load_data():
    df_refah = pd.read_csv('mash_cap_refah.csv')
    df = pd.read_csv('mofid_portfo.csv')
    return df_refah, df

df_refah, df = load_data()

# Function to format numbers
def format_number(x):
    if isinstance(x, (int, float)):  
        return f"{x:,.1f}"
    return x

# Bourse calculations
bourse = df_refah[df_refah['bourse'] > 10 * 1000 * 1000] 
bourse['bourse_mt'] = bourse['bourse'] / (10 * 1000 * 1000) 
bourse_g = bourse.groupby('city').agg(bourse_count=('id', 'count'), 
                                      median_mt=('bourse_mt', 'median'), 
                                      mean_mt=('bourse_mt', 'mean'), 
                                      p90_mt=('bourse_mt', lambda x: x.quantile(0.90)),
                                      p99_mt=('bourse_mt', lambda x: x.quantile(0.99))
                                     ).reset_index()

# Merging with reference table and formatting
df_refah_g = df_refah.groupby('city').agg(all_count=('id', 'count')).reset_index()                                 
bourse_g = pd.merge(df_refah_g[['city', 'all_count']], bourse_g, on='city', how='inner').sort_values('bourse_count', ascending=False)
columns_to_format = ['median_mt', 'mean_mt', 'p90_mt', 'p99_mt']
bourse_g[columns_to_format] = bourse_g[columns_to_format].applymap(format_number)

# Display Bourse data in the app
st.header("Data from Refah")
st.subheader("Bourse Data")
st.dataframe(bourse_g)

# Additional aggregations like `mandeh`, `card`, `car`.
mandeh = df_refah[~df_refah['mandeh_1400'].isna()] 
mandeh['mandeh_mt'] = mandeh['mandeh_1400'] / (10 * 1000 * 1000) 
mandeh_g = mandeh.groupby('city').agg(mandeh_count=('id', 'count'), 
                                      median_mt=('mandeh_mt', 'median'), 
                                      mean_mt=('mandeh_mt', 'mean'), 
                                      p90_mt=('mandeh_mt', lambda x: x.quantile(0.90)),
                                      p99_mt=('mandeh_mt', lambda x: x.quantile(0.99))
                                     )

mandeh_g = pd.merge(df_refah_g[['city', 'all_count']], mandeh_g, on='city', how='inner').sort_values('mandeh_count', ascending=False)
columns_to_format_2 = ['median_mt', 'mean_mt', 'p90_mt', 'p99_mt']
mandeh_g[columns_to_format_2] = mandeh_g[columns_to_format_2].applymap(format_number)

# Display Mandeh data
st.subheader("Mandeh Data")
st.dataframe(mandeh_g)

# Card data aggregation
card = df_refah[~df_refah['cardpermonth_1401'].isna()] 
card['card_mt'] = card['cardpermonth_1401'] / (10 * 1000 * 1000) 
card_g = card.groupby('city').agg(card_count=('id', 'count'), 
                                  card_median_mt=('card_mt', 'median'), 
                                  card_mean_mt=('card_mt', 'mean'), 
                                  card_p90_mt=('card_mt', lambda x: x.quantile(0.90)),
                                  card_p99_mt=('card_mt', lambda x: x.quantile(0.99))
                                  )

card_g = pd.merge(df_refah_g[['city', 'all_count']], card_g, on='city', how='inner').sort_values('card_count', ascending=False)
columns_to_format_3 = ['card_median_mt', 'card_mean_mt', 'card_p90_mt', 'card_p99_mt']
card_g[columns_to_format_3] = card_g[columns_to_format_3].applymap(format_number)

# Display Card Data
st.subheader("Card Data")
st.dataframe(card_g)

# Building car transactions statistics
car = df_refah[df_refah['carsprice'] > 0] 
car['car_mt'] = car['carsprice'] / (10 * 1000 * 1000)
car_g = car.groupby('city').agg(has_car_count=('id', 'count'), 
                                car_price_median_mt=('car_mt', 'median'), 
                                car_price_mean_mt=('car_mt', 'mean'), 
                                car_price_p90_mt=('car_mt', lambda x: x.quantile(0.90)),
                                car_price_p99_mt=('car_mt', lambda x: x.quantile(0.99))
                                ).reset_index()

car_g = pd.merge(df_refah_g[['city', 'all_count']], car_g, on='city', how='inner').sort_values('has_car_count', ascending=False)
columns_to_format_4 = ['car_price_median_mt', 'car_price_mean_mt', 'car_price_p90_mt', 'car_price_p99_mt']
car_g[columns_to_format_4] = car_g[columns_to_format_4].applymap(format_number)

# Display Car Data
st.subheader("Car Data")
st.dataframe(car_g)

# Building mofid
bourse_m = df[df['online_section'] > 10 * 1000 * 1000] # 1 Million Tomans in mandeh
bourse_m['bourse_mt'] = bourse_m['online_section'] / (10 * 1000 * 1000) # in Million Tomans
bourse_m_g = bourse_m.groupby('city').agg(mofid_count=('customerkey', 'count'), 
                                      mofid_median_mt=('bourse_mt', 'median'), 
                                      mofid_mean_mt=('bourse_mt', 'mean'), 
                                      mofid_p90_mt=('bourse_mt', lambda x: x.quantile(0.90)),
                                      mofid_p99_mt=('bourse_mt', lambda x: x.quantile(0.99))
                                     ).reset_index()
df_g = df.groupby('city').agg(all_count=('customerkey', 'count')).reset_index()                                 
bourse_m_g = pd.merge(df_g[['city', 'all_count']], bourse_m_g, on='city', how='inner').sort_values('mofid_count', ascending=False)
columns_to_format_5 = ['mofid_median_mt', 'mofid_mean_mt', 'mofid_p90_mt', 'mofid_p99_mt']
bourse_m_g[columns_to_format_5] = bourse_m_g[columns_to_format_5].applymap(format_number)

# Display mofid Data
st.header("Data from Mofid")
st.subheader("Online Section")
st.dataframe(bourse_m_g)

# Building mofid
bourse_m_mf = df[df['MF_section'] > 10 * 1000 * 1000] # 1 Million Tomans in mandeh
bourse_m_mf['bourse_mt'] = bourse_m_mf['MF_section'] / (10 * 1000 * 1000) # in Million Tomans
bourse_m_mf_g = bourse_m_mf.groupby('city').agg(mofid_count=('customerkey', 'count'), 
                                      mofid_median_mt=('bourse_mt', 'median'), 
                                      mofid_mean_mt=('bourse_mt', 'mean'), 
                                      mofid_p90_mt=('bourse_mt', lambda x: x.quantile(0.90)),
                                      mofid_p99_mt=('bourse_mt', lambda x: x.quantile(0.99))
                                     ).reset_index()
df_g = df.groupby('city').agg(all_count=('customerkey', 'count')).reset_index()                                 
bourse_m_mf_g = pd.merge(df_g[['city', 'all_count']], bourse_m_mf_g, on='city', how='inner').sort_values('mofid_count', ascending=False)
columns_to_format6 = ['mofid_median_mt', 'mofid_mean_mt', 'mofid_p90_mt', 'mofid_p99_mt']
bourse_m_mf_g[columns_to_format6] = bourse_m_mf_g[columns_to_format6].applymap(format_number)

# Display mofid Data
st.subheader("MF Section")
st.dataframe(bourse_m_mf_g)

st.subheader("AUM Histogram 1")
# Plotting AUM Histogram
# Plotting AUM Histogram 1
city_options = list(df['city'].unique())  # Get unique cities from your DataFrame

# Add a multi-select dropdown for city filtering (with unique key)
selected_city = st.multiselect('Select City', city_options, default=city_options, key='city_filter_1')

# Filter the DataFrame based on the selected city/cities
filtered_df = df[df['city'].isin(selected_city)]



fig, ax = plt.subplots(figsize=(10, 6))

# Filter x1 and x2 based on the filtered DataFrame
x1 = filtered_df[filtered_df['online_section'] > 10 * 1000 * 1000]
x2 = filtered_df[filtered_df['MF_section'] > 10 * 1000 * 1000]

# Plotting histograms
sns.histplot(np.log(x1['online_section'] / (10 ** 7) + 1), bins=50, edgecolor='white', stat='percent', kde=True, color='darkblue', alpha=0, ax=ax)
sns.histplot(np.log(x2['MF_section'] / (10 ** 7) + 1), bins=50, edgecolor='white', stat='percent', kde=True, color='darkgreen', alpha=0, ax=ax)

# Assuming 'bourse' is a separate DataFrame, apply the same city filter to it (if applicable)
filtered_bourse = bourse[bourse['city'].isin(selected_city)]
sns.histplot(np.log(filtered_bourse['bourse'] / (10 ** 7) + 1), bins=50, edgecolor='white', stat='percent', kde=True, color='darkred', alpha=0, ax=ax)

# Adding legends and labels to the graph
plt.xlabel('log(Amount)')
plt.ylabel('Percent')
plt.title('Histogram of AUM')
plt.legend(['Online', 'MF', 'Refah'])

# Show the plot in the Streamlit app
st.pyplot(fig)

st.subheader("AUM Histogram 2")
# Plotting AUM Histogram 2
selected_city2 = st.multiselect('Select City', city_options, default=city_options, key='city_filter_2')  # Unique key for the second multiselect

# Filter the DataFrame based on the selected city/cities
filtered_df2 = df[df['city'].isin(selected_city2)]



fig, ax = plt.subplots(figsize=(10, 6))

# Filter x1 and x2 based on the filtered DataFrame
x12 = filtered_df2[filtered_df2['online_section'] > 10 * 1000 * 1000]
x22 = filtered_df2[filtered_df2['MF_section'] > 10 * 1000 * 1000]

# Plotting histograms
sns.histplot(np.log(x12['online_section'] / (10 ** 7) + 1), bins=50, edgecolor='white', stat='percent', kde=True, color='darkblue', alpha=0, ax=ax)
sns.histplot(np.log(x22['MF_section'] / (10 ** 7) + 1), bins=50, edgecolor='white', stat='percent', kde=True, color='darkgreen', alpha=0, ax=ax)

# Assuming 'bourse' is a separate DataFrame, apply the same city filter to it (if applicable)
filtered_bourse2 = bourse[bourse['city'].isin(selected_city2)]
sns.histplot(np.log(filtered_bourse2['bourse'] / (10 ** 7) + 1), bins=50, edgecolor='white', stat='percent', kde=True, color='darkred', alpha=0, ax=ax)

# Adding legends and labels to the graph
plt.xlabel('log(Amount)')
plt.ylabel('Percent')
plt.title('Histogram of AUM')
plt.legend(['Online', 'MF', 'Refah'])

# Show the plot in the Streamlit app
st.pyplot(fig)