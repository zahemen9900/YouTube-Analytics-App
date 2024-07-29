import streamlit as st
from streamlit_option_menu import option_menu

import streamlit as st
from typing import Union
import os 
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

#to process mail and calls
import time
import os
import re
import requests
import smtplib
from email.mime.text import MIMEText
from warnings import simplefilter
from googleapiclient.discovery import build

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import get_csv_from_loc, extract_channel_info, load_objects, predict_cluster, generate_recommendations, deliver_recommendations, give_feedback
# For Navigation Menu and page configurations


# Note: I used `print()` for exception handling instead of `st.write()` or `st.error()` since the user does not need to see the logs of those errors
    
country_abbreviations = {
    'Unknown': 'Unknown', 'US': 'United States', 'IN': 'India', 'BR': 'Brazil', 'MX': 'Mexico', 'RU': 'Russia',
    'PK': 'Pakistan', 'PH': 'Philippines', 'ID': 'Indonesia', 'TH': 'Thailand', 'FR': 'France', 'CO': 'Colombia',
    'IQ': 'Iraq', 'JP': 'Japan', 'EC': 'Ecuador', 'AR': 'Argentina', 'TR': 'Turkey', 'SA': 'Saudi Arabia',
    'SV': 'El Salvador', 'BD': 'Bangladesh', 'GB': 'United Kingdom', 'DZ': 'Algeria', 'ES': 'Spain', 'PE': 'Peru',
    'EG': 'Egypt', 'JO': 'Jordan', 'MA': 'Morocco', 'SG': 'Singapore', 'SO': 'Somalia', 'CN': 'China', 'CA': 'Canada',
    'AU': 'Australia', 'KR': 'South Korea', 'DE': 'Germany', 'NG': 'Nigeria', 'ZA': 'South Africa', 'IT': 'Italy',
    'VN': 'Vietnam', 'NL': 'Netherlands', 'CL': 'Chile', 'MY': 'Malaysia', 'GR': 'Greece', 'SE': 'Sweden',
    'CH': 'Switzerland', 'AT': 'Austria', 'NO': 'Norway', 'DK': 'Denmark', 'NZ': 'New Zealand', 'IE': 'Ireland',
    'PT': 'Portugal', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'PL': 'Poland', 'RO': 'Romania', 'UA': 'Ukraine',
    'BE': 'Belgium', 'AZ': 'Azerbaijan', 'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan', 'IL': 'Israel', 'IS': 'Iceland',
    'FI': 'Finland', 'FJ': 'Fiji', 'PG': 'Papua New Guinea', 'SB': 'Solomon Islands', 'VU': 'Vanuatu', 'TO': 'Tonga',
    'WS': 'Samoa', 'TV': 'Tuvalu', 'KI': 'Kiribati', 'MH': 'Marshall Islands', 'PW': 'Palau', 'FM': 'Micronesia',
    'NR': 'Nauru', 'TL': 'East Timor',
}
    
continent_mapping = {
    'Unknown': 'Unknown',
    'United States': 'North America', 'India': 'Asia', 'Brazil': 'South America', 'Mexico': 'North America',
    'Russia': 'Europe', 'Pakistan': 'Asia', 'Philippines': 'Asia', 'Indonesia': 'Asia', 'Thailand': 'Asia',
    'France': 'Europe', 'Colombia': 'South America', 'Iraq': 'Asia', 'Japan': 'Asia', 'Ecuador': 'South America',
    'Argentina': 'South America', 'Turkey': 'Asia', 'Saudi Arabia': 'Asia', 'El Salvador': 'North America',
    'Bangladesh': 'Asia', 'United Kingdom': 'Europe', 'Algeria': 'Africa', 'Spain': 'Europe', 'Peru': 'South America',
    'Egypt': 'Africa', 'Jordan': 'Asia', 'Morocco': 'Africa', 'Singapore': 'Asia', 'Somalia': 'Africa',
    'China': 'Asia', 'Canada': 'North America', 'Australia': 'Oceania', 'South Korea': 'Asia', 'Germany': 'Europe',
    'Nigeria': 'Africa', 'South Africa': 'Africa', 'Italy': 'Europe', 'Vietnam': 'Asia', 'Netherlands': 'Europe',
    'Chile': 'South America', 'Malaysia': 'Asia', 'Greece': 'Europe', 'Sweden': 'Europe', 'Switzerland': 'Europe',
    'Austria': 'Europe', 'Norway': 'Europe', 'Denmark': 'Europe', 'New Zealand': 'Oceania', 'Ireland': 'Europe',
    'Portugal': 'Europe', 'Czech Republic': 'Europe', 'Hungary': 'Europe', 'Poland': 'Europe', 'Romania': 'Europe',
    'Ukraine': 'Europe', 'Belgium': 'Europe', 'Azerbaijan': 'Asia', 'Kazakhstan': 'Asia', 'Uzbekistan': 'Asia',
    'Israel': 'Asia', 'Iceland': 'Europe', 'Finland': 'Europe', 'Argentina': 'South America', 'Brazil': 'South America',
    'Colombia': 'South America', 'Mexico': 'North America', 'Peru': 'South America', 'Venezuela': 'South America',
    'Cuba': 'North America', 'Jamaica': 'North America', 'Honduras': 'North America', 'Nicaragua': 'North America',
    'Panama': 'North America', 'Guatemala': 'North America', 'Costa Rica': 'North America', 'Bolivia': 'South America',
    'Paraguay': 'South America', 'Uruguay': 'South America', 'Guyana': 'South America', 'Suriname': 'South America',
    'French Guiana': 'South America', 'Ecuador': 'South America', 'Chile': 'South America', 'Fiji': 'Oceania',
    'Papua New Guinea': 'Oceania', 'Solomon Islands': 'Oceania', 'Vanuatu': 'Oceania', 'Tonga': 'Oceania',
    'Samoa': 'Oceania', 'Tuvalu': 'Oceania', 'Kiribati': 'Oceania', 'Marshall Islands': 'Oceania', 'Palau': 'Oceania',
    'Micronesia': 'Oceania', 'Nauru': 'Oceania', 'East Timor': 'Asia'
}

popular_countries = [
    'United States', 'India', 'Brazil', 'Mexico', 'Russia', 'Pakistan', 'Philippines', 'Indonesia',
    'Thailand', 'France', 'Colombia', 'Iraq', 'Japan', 'Ecuador', 'Argentina', 'Turkey', 'Saudi Arabia',
    'El Salvador', 'Bangladesh', 'United Kingdom', 'Algeria', 'Spain', 'Peru', 'Egypt', 'Jordan', 'Morocco',
    'Singapore', 'Somalia', 'Canada', 'Germany', 'Italy', 'South Korea', 'Australia', 'Netherlands', 'Chile',
    'South Africa', 'Vietnam', 'Malaysia', 'Israel', 'Belgium', 'Sweden', 'Switzerland', 'Austria', 'Greece',
    'Norway', 'Denmark', 'Poland', 'Ireland', 'Portugal', 'Ukraine', 'India', 'Brazil', 'Mexico', 'Russia',
    'Pakistan', 'Philippines', 'Indonesia', 'Thailand', 'France', 'Colombia', 'Iraq', 'Japan', 'Ecuador',
    'Argentina', 'Turkey', 'Saudi Arabia', 'El Salvador', 'Bangladesh', 'United Kingdom', 'Algeria', 'Spain',
    'Peru', 'Egypt', 'Jordan', 'Morocco', 'Singapore', 'Somalia', 'Nigeria', 'Kenya', 'Ghana', 'South Africa',
    'Ethiopia', 'Uganda', 'Tanzania', 'Malawi', 'Zimbabwe', 'Zambia', 'Mozambique', 'Angola', 'Congo', 'Niger',
    'Mali', 'Mauritania', 'Senegal', 'Benin', 'Burkina Faso', 'Sierra Leone', 'Liberia', 'Guinea', 'Togo'
]



st.session_state['channel_name'] = ''

st.set_page_config(
    page_title = 'YouTube Analytics App',
    page_icon = '🌟',
    initial_sidebar_state = 'collapsed'

    )

selected = option_menu(

    menu_title = None,
    options = ['App Home', 'Summary Stats', 'Top YouTubers in Categories', 'Your Recommendations', 'ML Highlights / About Project'],
    icons = ['cast', 'speedometer', 'stars', 'blockquote-left','info-circle'],
    default_index = 0,
    orientation = 'horizontal'

    )


if not all([obj in st.session_state for obj in ['pipeline_l', 'cpm_scaler_l', 'model_l', 'tokenizer_l']]):
    st.session_state['pipeline_l'], st.session_state['cpm_scaler_l'], st.session_state['model_l'], st.session_state['tokenizer_l'] = load_objects('objects')

for obj in [st.session_state['pipeline_l'], st.session_state['cpm_scaler_l'], st.session_state['model_l'], st.session_state['tokenizer_l']]:
    st.write(type(obj))


def main():


    data = get_csv_from_loc()

    with st.sidebar:
        st.markdown(
            """<h1 style = "font-size: 45px; font-family: Arial, sans-serif;">Feedback<h1>"""
            , unsafe_allow_html = True)

        feedback_menu = give_feedback()



    # For Home Page Section;
    if selected == 'App Home':
        col1, col2 = st.columns([.7, .3])

        title = col1.markdown(
            """
            # <div style = "font-size: 80px; font-family: Arial, sans-serif; text-align: left;"><b>YouTube Channel Tip App</b></div>
            """, unsafe_allow_html = True
        )
        yt_icon = col2.markdown(
            """
            <div style = "text-align: top;">
            <img src = "https://cdn-icons-png.flaticon.com/256/1384/1384060.png" width = "320" height = "300" alt = "YouTube Logo"></div>
            """
            , unsafe_allow_html = True
        )
        st.markdown(
            """
            <div style = "padding: 20px;"></div>
            """
            , unsafe_allow_html = True
        )

        description = st.markdown(
            """
            <div style = "font-family: Arial, sans-serif">
            <b><p style = "font-size: 20px;">Do you want to take your YouTube Channel to the next level, but don't know where to start?💭🤔</p><p style = "font-size: 20px;"> You've come to the right place! Here you can get personalized recommendations to boost your likes, subscribers, and visits.⚡📈</p></b>

            <p></p><p></p>
            <p style = "font-size: 20px;">Our app is powered by data from the YouTube API and a machine learning model that analyzes the performance of different types of channels. We will show you some of our insights on the <b style = "color: brown;">YouTube Channel Analytics</b> and how they can help you improve your channel.</p>

            <p style = "font-size: 20px;">Our app will guide you through:</p>
            <ul style = "font-size: 20px;">
            <li>Understanding the data that was used for making recommendations.</li>
            <li>Getting your customized recommendations based on your channel category and goals.</li>
            <li>Understanding the magic behind the predictions and how it works.</li>
            </ul>

            <p style = "font-size: 22px;">Are you ready to grow your channel? Let's get started!✅</p>
            </div>

            <p></p><p></p>
            """,
            unsafe_allow_html=True
        )


    if selected == 'Summary Stats':
        st.write("** _**Hover over plots to reveal info!**_")
        #Country Distribution
        try:
            country_data = data['Country'].value_counts()

            least_10 = country_data.nsmallest(10)

            country_data.drop(least_10.index, inplace = True)

            country_data['Other'] = least_10.sum()

            #country_data = country_data.reset_index().rename(columns = {
            #   'Country': 'Count',
            #   'index': 'Country'
            #   })

            fig = px.pie(country_data.reset_index(), values = 'count', names = 'Country', color_discrete_sequence = px.colors.diverging.Spectral)

            fig.update_layout(
                title_text='<b style = "font-family: Arial, sans-serif">Country Distributions in Data</b>',
                title_font=dict(size=30, family='Arial'),
                title=dict(x=0.5, xanchor='center'),
                autosize = True,
                width = 650,
                height = 500
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.write(f'An error occured: {e}')


        st.write("**How does each Country perform Metric-wise?** 💭")
        #For barplot
        try:
            metric_avgs = data.groupby('Continent')[['Subscribers', 'Visits', 'Likes']].mean().reset_index()

            metric_avgs.rename(columns={
                'Subscribers': 'Average Subscribers per Channel',
                'Visits': 'Average Visits per Channel',
                'Likes': 'Average Likes per Video'
            }, inplace=True)

            fig3 = sp.make_subplots(1, 3)

            fig3.add_trace(go.Bar(x=metric_avgs['Continent'], y=metric_avgs['Average Subscribers per Channel'], marker=dict(color=px.colors.sequential.Inferno)), row=1, col=1)
            fig3.add_trace(go.Bar(x=metric_avgs['Continent'], y=metric_avgs['Average Visits per Channel'], marker=dict(color=px.colors.sequential.Plasma)), row=1, col=2)
            fig3.add_trace(go.Bar(x=metric_avgs['Continent'], y=metric_avgs['Average Likes per Video'], marker=dict(color=px.colors.sequential.Inferno)), row=1, col=3)

            fig3.update_xaxes(title_text='<b style = "font-family: Arial, sans-serif;">Average Subscribers per Channel</b>', row=1, col=1)
            fig3.update_xaxes(title_text='<b style = "font-family: Arial, sans-serif;">Average Visits per Channel</b>', row=1, col=2)
            fig3.update_xaxes(title_text='<b style = "font-family: Arial, sans-serif;">Average Likes per Video</b>', row=1, col=3)

            fig3.update_layout(title_text='<b style = "font-family: Arial, sans-serif;">Metric Performances across Continents</b>',
                        title_font=dict(size=30, family='Arial'),
                        title=dict(x=0.5, xanchor='center'))
            
            fig3.update_layout(
                autosize=True,
                width = 750,
                height = 500,
                showlegend = False
            )

            st.plotly_chart(fig3)

        except Exception as e:
            st.write(f'Error: {e}')



        st.write('**What are the different categories of YouTube Channels?** 🫧')
        # Scatter Matrix
        try:

            dd = data[['Subscribers', 'Likes', 'Visits', 'Cluster']].copy()

            dd['Cluster'] = dd['Cluster'].apply(lambda x: str(x))

            fig2 = ff.create_scatterplotmatrix(dd,
                                              diag='box', index ='Cluster',)

            fig2.update_layout(
                autosize = False,
                width = 650,
                height = 1000,
            )
            fig2.update_layout( title_text = '<b style = "font-family: Arial, sans-serif">Metric Correlations & Distributions</b>', 
                              title_font = dict(size = 30, family = 'cooper black'),
                               title = dict(x = 0.5, xanchor = 'center')
                
            )
        except Exception as e:
            st.write(f'An error occured: {e}')

        st.plotly_chart(fig2)

        st.markdown(
            """
            <div style="border-color: #d3d3d3; border-width: 4px; border-style: solid; border-radius: 10px; padding: 15px; margin: 10px; font-family: Arial, Helvetica, sans-serif; box-shadow: 5px 5px 10px grey;">
              <h3 style="color: gray; margin-bottom: 10px;"><b>A Detailed Exploration of Channel Categories 🔍</b></h3>
              
              <h5 style="color: blue;"><b>Rank 3: Rising Stars💫</b></h5> 
              <p>In this category, channels emerge with fewer visits, likes, and subscribers, steadily climbing the ranks of Top YouTubers. They embody the aspiring talents, on the verge of breaking into the mainstream.</p>
              
              <h5 style="color: #FFA500;"><b>Rank 0: Ground Zero📉</b></h5> 
              <p>This category represents the YouTubers on the lowest end of the spectrum. Among the top YouTubers, they fall behind the most in all the respective metrics, and are also very populous.</p>
              
              <h5 style="color: #008000;"><b>Rank 1: Subscribers' Haven✨</b></h5> 
              <p>This category hosts channels with a substantial subscriber base but modest likes and visits. Recognized for their high retention rates, they craft popular content, albeit at a less frequent pace, resulting in a distinct engagement pattern.</p>
              
              <h5 style="color: #FF0000;"><b>Rank 4: Balancing Act 🦾</b></h5> 
              <p>Moderate in visits, likes, and subscribers, these channels strike a balance on the lower spectrum, outshining <b style="color: #FFA500;">Category 2</b> in overall metrics. They hold a middle ground, contributing to the diverse YouTube landscape.</p>
              
              <h5 style="color: #800080;"><b>Rank 2: Engaging Echoes🔊</b></h5> 
              
              <p style = "color:default;">Channels in this category boast the highest likes and visits, yet maintain a humble subscriber count. They epitomize high engagement but wrestle with retention rates, creating a vibrant but fleeting viewership.</p>
            </div>

            <p></p>

            """,
            unsafe_allow_html=True
        )


        #Cluster concentrations in Different Continents

        st.write('##### _**How are the Clusters Distributed by Continent?**_ 🌍')

        try:
            clusters = data['Cluster'].unique()

            color_sets = [px.colors.sequential.Magma, px.colors.sequential.Cividis, px.colors.sequential.Viridis, px.colors.diverging.Spectral]

            for cluster in clusters:
                data_ = data.loc[data['Cluster'] == cluster]

                country_dist = data_['Country'].value_counts().reset_index()


                fig4 = px.pie(country_dist, values = 'count', names = 'Country',
                            color_discrete_sequence = color_sets[cluster % len(color_sets)])

                fig4.update_layout(title_text='<b style = "font-family: Arial, sans-serif">Country Distribution in Cluster {}</b>'.format(cluster),
                            title_font=dict(size=30, family='Arial'),
                                title = dict(x = 0.5, xanchor = 'center'))

                fig4.update_layout(autosize = True)

                st.plotly_chart(fig4)

        except Exception as e:
            st.write(f'An error occured: {e}')


        st.title("**Other Important Metrics**")
        with st.expander('**Expand to see Continent Preferrences Globally**'):
            try:
                dataforplot = data['Categories'].value_counts().reset_index()
                dataforplot['Categories'] = dataforplot['Categories'].str.replace('Salud y autoayuda', 'Health and self-help')
                fig6 = px.pie(dataforplot, values = 'count', names = 'Categories', color_discrete_sequence = px.colors.diverging.PRGn)
                fig6.update_layout(title_text='<b style = "font-family: Arial, sans-serif;">Category Distributions</b>',
                              title_font=dict(size=30, family='Arial'),
                                    title = dict(x = 0.5, xanchor = 'center'))
                fig6.update_layout(autosize = False,
                    width = 650, height = 500)

                st.plotly_chart(fig6)

            except Exception as e:
                st.write(f'Error: {e}')




        with st.expander("**Expand to see Which content are preferred across Continents** 🌐"):
            try:
                color_sets = [px.colors.sequential.Magma, px.colors.sequential.Cividis, px.colors.sequential.Viridis]
                continents = data['Continent'].unique().tolist()
                for continent in continents:
                    if continent == 'Unknown':
                        continue
                    relevant_data = data.loc[data.Continent == continent]
                    relevant_data['Categories'] = relevant_data['Categories'].str.replace('Salud y autoayuda', 'Health and self-help')
                    cluster_proportions = relevant_data['Categories'].value_counts().reset_index()

                    fig7 = px.bar(cluster_proportions, x = 'count', y = 'Categories', color_discrete_sequence = color_sets[continents.index(continent) % len(color_sets)])

                    fig7.update_layout(title_text=f'<b style = "font-family: Arial, sans-serif">Categories in {continent}</b>',
                                title_font=dict(size=25, family='Arial'),
                                title=dict(x=0.5, xanchor='center'),
                                                autosize=False, width = 650, height = 420)

                    st.plotly_chart(fig7)

            except Exception as e:
                st.write(e)
                
    if selected == 'Top YouTubers in Categories':
        st.markdown(
        """
        <style>
            .channel-name {
                color: gray; 
            }
            .rounded-images {
                border-radius: 15px;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.5);
                overflow: hidden;
                margin-bottom: 20px;
            }
    
            div {
                font-family: Arial, sans-serif;
            }
    
        </style>
    
        <div>
            <h2 class="channel-name"><b>PewDiePie & Mr Beast (Rank 1)</b></h2>
            <p>PewDiePie is a Swedish YouTuber who is known for his gaming videos, comedy sketches, and meme reviews. He is <b>the most-subscribed individual creator</b> on YouTube with over <b>110 million subscribers</b>. Mr Beast is an American YouTuber who is famous for his expensive stunts, philanthropic acts, and viral challenges. He has over <b>80 million subscribers</b> and is one of the highest-earning YouTubers in the world.</p>
            <div class="rounded-images">
                <img src="https://hips.hearstapps.com/hmg-prod/images/pewdiepie_gettyimages-501661286.jpg?resize=1200:*" alt="PewDiePie" width="350" height="350">
                <img src="https://wallpapers.com/images/hd/mr-beast-bright-screen-background-he6y102ildr4ca8q.jpg" alt="Mr Beast" width="349" height="350">
            </div>
            <p></p><p></p>
        </div>
    
        <div>
            <h2 class="channel-name"><b>The Ellen Show & Katy Perry (Rank 4)</b></h2>
            <p>The Ellen Show is an American daytime television variety comedy talk show hosted by Ellen DeGeneres. It has been running for <b>19 seasons</b> and has won numerous awards, including 11 Daytime Emmys for Outstanding Talk Show Entertainment. Katy Perry is an American singer, songwriter, and television personality. She is one of the best-selling music artists of all time, with over <b>143 million records sold worldwide</b>. She has nine U.S. number one singles and has received various accolades, including five American Music Awards and a Brit Award</p>
            <div class="rounded-images">
                <img src="https://m.media-amazon.com/images/M/MV5BODA5ZDQyMzYtZWQwMy00MDQ1LWE2OGUtNGYyNTk0Y2NhZGM4XkEyXkFqcGdeQXVyMTkzODUwNzk@._V1_.jpg" alt="The Ellen Show" width="350" height="450">
                <img src="https://m.media-amazon.com/images/M/MV5BMjE4MDI3NDI2Nl5BMl5BanBnXkFtZTcwNjE5OTQwOA@@._V1_.jpg" alt="Katy Perry" width="349" height="450">
            </div>
            <p></p><p></p>
        </div>
    
        <div>
            <h2 class="channel-name"><b>Techno Gamers & Kimberly Loaiza (Rank 2)</b></h2>
            <p>Techno Gamers is an Indian gaming YouTuber who creates videos of gameplays and live streams of <b>GTA 5</b>, <b>Minecraft</b>, <b>Call of Duty</b>, and more. He has <b>over 37 million subscribers</b> and is one of the most popular gamers in India. Kimberly Loaiza is a Mexican internet personality and singer who started her YouTube career in 2016. She is currently the seventh most-followed user on TikTok and has over <b>40 million subscribers</b> on YouTube. She also has a music career and has released several singles, such as <em><b>Enamorarme</b>, <b>Patán</b></em>, and <em><b>Kitty</b></em>.</p>
            <div class="rounded-images">
                <img src="https://img.gurugamer.com/resize/740x-/2021/04/02/youtuber-ujjwal-techno-gamerz-3aa0.jpg" alt="Techno Gamerz" width="350" height="450">
                <img src="https://m.media-amazon.com/images/I/71G48FB73WL._AC_UF1000,1000_QL80_.jpg" alt="Kimberly Loaiza" width="349" height="450">
            </div>
            <p></p><p></p>
        </div>
    
        <div>
            <h2 class="channel-name"><b>SSSniperWolf & JackSepticEye (Rank 3)</b></h2>
            <p>SSSniperWolf is a British-American YouTuber who is known for her gaming and reaction videos. She has over <b>30 million subscribers</b> and is one of the most-watched female gamers on YouTube. JackSepticEye is an Irish YouTuber who is also known for his gaming and vlog videos. He has over <b>27 million subscribers</b> and is one of the most influential Irish online personalities. He has also appeared in the film Free Guy and released a biographical documentary called <b><em>How Did We Get Here?</em></b></p>
            <div class="rounded-images">
                <img src="https://ih1.redbubble.net/image.2189561281.9428/mwo,x1000,ipad_2_skin-pad,750x1000,f8f8f8.u1.jpg" alt="SSSniper Wolf" width="350" height="420">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Jacksepticeye_by_Gage_Skidmore.jpg/1200px-Jacksepticeye_by_Gage_Skidmore.jpg" alt="JackSepticEye" width="349" height="420">
            </div>
            <p></p><p></p>
        </div>
    
        <div>
            <h2 class="channel-name"><b>JessNoLimit & Daddy Yankee (Rank 0)</b></h2>
            <p>JessNoLimit is an Indonesian gaming YouTuber and Instagram star who is known for his Mobile Legends gameplays. He has over <b>42 million subscribers</b> and is the <b>third most-subscribed YouTuber in Indonesia</b>. Daddy Yankee is a Puerto Rican rapper, singer, songwriter, and actor who is considered the <b><em>"King of Reggaeton"</em></b>. He has sold over <b>30 million records worldwide</b> and has won numerous awards, including five Latin Grammy Awards and two Billboard Music Awards. He is best known for his hit songs like <em><b>Gasolina</b>, <b>Despacito</b></em>, and <em><b>Con Calma</b></em>.</p>
            <div class="rounded-images">
                <img src="https://akcdn.detik.net.id/visual/2023/05/05/jess-no-limit-dan-sisca-kohl-2_43.png?w=650&q=90" alt="JessNoLimit" width="350" height="300">
                <img src="https://people.com/thmb/eT6A-wncUzuDs-XV08qRSd_gSUk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(688x281:690x283)/Daddy-Yankee-Retirement-120623-a855484297944821ad14c8b98453b6a5.jpg" alt="Daddy Yankee" width="349" height="300">
            </div>
        </div>
    
        """,
        unsafe_allow_html=True
        )

        st.write("_** Ranked based on Popularity_")

    if selected == 'ML Highlights / About Project':
        st.title(
            """
            **The Magic Behind the Scenes ✨**
            
            ###### _(How We Trained and Validated Our Model)_
            ---

            """
        )

        st.markdown(
            """
            <div style = "font-family: Arial, sans-serif;"><p>You might be curious about how we're generating the awesome recommendations for you. Well, the secret is <b>a powerful machine learning model</b> from the <code>Scikit-Learn</code> library, and with some clever hyperparameter-tuning and other techniques, we achieved some amazing results! 🙌</p>
            <p>Here are some highlights from the training process:</p><div>
            """, unsafe_allow_html = True
        )

        st.write('---')


        col1, col2 = st.columns([.5, .5])
        st.write('---')
        col1.write("#### Training & Evaluation Model:")
        col2.write(st.session_state['model_l'].summary())

        st.write("_**A glace at the dataset:**_")

        table = st.write(data.head(10))

        st.write("See the full dataset [here](https://github.com/zahemen9900/YouTube-Analytics-App/blob/main/YouTube%20Data%20EDA/yt_cluster_data.csv)")
        st.write("Also see full details of model in [this notebook](https://colab.research.google.com/drive/1iEWco4Cu_2D4HbVk0lVCM_nl3mUVOvyW?usp=sharing)")

        st.markdown(
            """
            <p></p>
            <div style="font-size: 15px; font-family: Arial, sans-serif">
            <h5><b>More on Data Used</b></h5>

            <p>As an aspiring Top YouTuber, you deserve to compare yourself with the best of the best. That's why we used a curated dataset of the world's top 1000 YouTubers, to give you accurate assessments and useful tips to level up your game. We hope you enjoyed the recommendation as much as we enjoyed making this project 😊</p>

            <p>If you have any suggestions or recommendations, feel free to leave them in the <b>side-bar to your left ↖️</b>, and I'd really appreciate it</p>

            <p>If you are curious about how we made this app possible, or want to explore more resources for the project, you can check out <b>Zahemen's GitHub</b> <a href = "https://github.com/zahemen9900/YouTube-Analytics-App">here</a> and <a href = "https://github.com/zahemen9900/Analytics_App_YT">here</a> . You will find the source code, the data, and more.</p>
            <p>Leave a star on my GitHub if you enjoyed using the app, and thank you for your time and attention!</p></div>

            """,
            unsafe_allow_html=True
        )



    if selected == 'Your Recommendations':

        if 'recs' not in st.session_state:
            st.session_state['recs'] = None


        if 'recommendations' not in st.session_state:
            st.session_state['recommendations'] = None

        st.header("Enter your channel link below")

        with st.form(key = 'channel-url-form'):
            yt_url = st.text_input(label = '**Channel URL goes here** _( please remove any quotation marks from URL)_')

            st.write("Not sure how to get the link? Visit [this page](https://www.wikihow.com/Find-Your-YouTube-URL)")


            st.write("Try any of these as examples (just copy into link section):")

            st.write({
                'channel_url': 'www.youtube.com/channel/UCBJycsmduvYEL83R_U4JriQ',
                'category': 'Technology'
            })

            '_**or**_'
            
            st.write({
                'channel_url': 'https://www.youtube.com/channel/UCv8G-xZ_BBGufjbN6jbvLMQ',
                'category': 'Comedy'
            })

            selected_cat = st.radio("#### **Choose your channel category**", (
                    "Animation", "Toys", "Movies", "Video Games", "Music and Dance",
                    "News and Politics", "Fashion", "Animals and Pets", "Education",
                    "DIY and Life Hacks", "Fitness", "ASMR", "Comedy", "Technology", "Automobiles",
                    "Food and Cooking", "Travel and Adventure", "Science and Technology",
                    "Health and Wellness", "History and Documentaries", "Lifestyle Content",
                    "Book Reviews", "Art and Creativity", "Home Decor", "Parenting", "Business and Finance",
                    "Social Issues", "Photography", "Spirituality", "Language Learning",
                    "Sports and Athletics", "Entertainment News", "Pop Culture", "Podcasts"))

            submit_btn = st.form_submit_button('Submit', help = 'Submit to see your channel ')

            if submit_btn:
                if not (yt_url.startswith('https://www.youtube.com/channel/') or yt_url.startswith('www.youtube.com/channel/') or yt_url.startswith('https://youtube.com/channel/') or yt_url.startswith('youtube.com/channel/')):
                    st.error('Please enter a valid channel URL. Also make sure your link does not contain quotation marks')
                    st.session_state['recs'] = None
                else:
                    st.success('Url received!')
                    with st.spinner('Retrieving channel info...'):
                        time.sleep(1.5)

                    new_channel_df = extract_channel_info(yt_url, selected_cat)
                    if isinstance(new_channel_df, str):
                        st.session_state['recs'] = None
                        
                    with st.spinner('Getting your recommendations...'):
                        time.sleep(1.5)

                    recs = generate_recommendations(
                        yt_channel_df, st.session_state['model_l'], st.session_state['pipeline_l'], st.session_state['cpm_scaler_l'], st.session_state['tokenizer_l']
                    )
                    
                    st.write(recs)
                    st.session_state['recs'] = recs
        
        if st.session_state['recs'] is not None:
            rc_ = deliver_recommendations('form_auto', st.session_state['recs'])



        st.write("### Or enter your details manually below")


        with st.expander("**Expand to enter your info manually**"):
            if "formbtn_state" not in st.session_state:
                st.session_state.formbtn_state = False

            if st.session_state.formbtn_state:
                st.session_state.formbtn_state = True

            with st.form(key="channel_form"):

                col_01, col_02 = st.columns([.5, .5])

                channel_name = col_01.text_input('What is your channel name?', 'eg. Uncanny Valley')
                st.session_state['channel_name'] = channel_name

                selected_country = col_02.selectbox("Select your country", popular_countries)

                default_continent = continent_mapping.get(selected_country, 'Unknown')

                continents = data['Continent'].unique()
                continents = [continent for continent in continents]

                # Get the index of the default continent in the list of continents
                # If the default continent is not in the list, use 0 as the index
                default_index = continents.index(default_continent) if default_continent in continents else 0

                selected_continent = col_01.selectbox("Select your Continent", continents, index = default_index)

                n_visits = col_02.text_input('How many Visits do you have on average?')

                n_likes = col_01.text_input('How many likes do you get on average?')

                n_subs = col_02.text_input('How many subscribers do you have?')

                st.write("Select the category of your videos:")
                selected_category = st.radio("Options", (
                    "Animation", "Toys", "Movies", "Video Games", "Music and Dance",
                    "News and Politics", "Fashion", "Animals and Pets", "Education",
                    "DIY and Life Hacks", "Fitness", "ASMR", "Comedy", "Technology", "Automobiles",
                    "Food and Cooking", "Travel and Adventure", "Science and Technology",
                    "Health and Wellness", "History and Documentaries", "Lifestyle Content",
                    "Book Reviews", "Art and Creativity", "Home Decor", "Parenting", "Business and Finance",
                    "Social Issues", "Photography", "Spirituality", "Language Learning",
                    "Sports and Athletics", "Entertainment News", "Pop Culture", "Podcasts"
                ))

                submit_button = st.form_submit_button(label="Submit", help = 'Submit to see your recommendations!')

                if submit_button and not any(metric is None for metric in [n_visits, n_likes, n_subs]):
                    try:
                        n_visits, n_likes, n_subs = int(n_visits), int(n_likes), int(n_subs)
                    except:
                        st.error("Couldn't proceed. Please make sure that visits, likes and subscribers are all entered and are all numeric")
                        submit_button = False

                if submit_button:
                    st.success('Form submitted, Results are underway!')
                    time.sleep(1)

                    with st.spinner('Loading recomendations'):
                        time.sleep(1)

                    user_input =  pd.DataFrame({'Username' : channel_name,
                            'Categories': selected_category,
                            'Subscribers': n_subs,
                            'Country': selected_country,
                            'Continent' : selected_continent,
                            'Visits' : n_visits,
                            'Likes' : n_likes
                            }, index = [0])


                    personalized = generate_recommendations(
                        user_input, st.session_state['model_l'], st.session_state['pipeline_l'], st.session_state['cpm_scaler_l'], st.session_state['tokenizer_l']
                    )


                    st.write(personalized)

                    st.session_state['recommendations'] = personalized

        if st.session_state['recommendations'] is not None:
            rc = deliver_recommendations('form_manual', st.session_state['recommendations'])



if __name__ == '__main__':
    main()
