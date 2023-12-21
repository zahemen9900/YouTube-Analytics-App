import time
import os
import requests
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from warnings import simplefilter
from googleapiclient.discovery import build


simplefilter('ignore')

@st.cache_data
def get_csv_from_url(url = r"https://raw.githubusercontent.com/zahemen9900/YouTube-Analytics-App/main/YouTube%20Data%20EDA/yt_cluster_data.csv"):
    df = pd.read_csv(url)
    return df


# Defining plotting functions

sns.set_style('dark')
plt.rc('figure', autolayout=True)
plt.rc(
    'axes',
    labelweight='bold',
    labelsize=10,
    titleweight='bold',
    titlesize=14,
    titlepad=10,
)

@st.cache_data()
def pie_plot():
    plt_data = data['Country'].value_counts()

    least_22 = data['Country'].value_counts().nsmallest(22)
    plt_data.drop(least_22.index, inplace=True)
    plt_data['Other'] = least_22.sum()
    n  = len(plt_data)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig = plt.pie(plt_data.values, labels = plt_data.index,
                  autopct='%1.2f%%', shadow = True,
                  colors=sns.color_palette('coolwarm', n_colors=len(plt_data)), startangle=90,
                   explode = tuple([0.1] + (n-1) * [0]))

    ax.set_title('Distribution of Countries in Dataset', fontweight='bold')
    st.pyplot(plt.gcf())


@st.cache_data()
def preferences_across_continents(df: pd.core.frame.DataFrame):
    df_c = df.copy()

    try:
        if 'Continent' in df_c.columns:
            continent_names = df_c['Continent'].unique()

            continent_names = [name for name in continent_names if name != 'Unknown']

            n = int(len(continent_names))
            fig, axes = plt.subplots(2 * int(n / 2), 1, figsize=(3 * n, 3 * n))
            
            for continent, ax in zip(continent_names, axes.flat):

                target_data = df_c.loc[df_c['Continent'] == continent]
                category_counts = target_data['Categories'].value_counts()
    
                # Sieving out the smaller categories for a more structured plot
                
                k  = target_data['Categories'].nunique()
                if k > 10:
                    idx_to_drop = int(k / 4) * 3
                    least_cat = category_counts.nsmallest(idx_to_drop)
                    category_counts.drop(least_cat.index, inplace = True)
                    category_counts['Other'] = least_cat.sum()   

                n = len(category_counts)           

                ax.pie(category_counts.values, labels=category_counts.index, autopct = '%1.1f%%', startangle = 90,
                        colors = sns.color_palette('Spectral', len(category_counts) + 1), shadow = True, 
                        explode = tuple([0.1] + (n-1) * [0]))
                ax.set_title('Genre Popularity in {}'.format(continent))

            st.pyplot(plt.gcf())

    except Exception as e:
        return e

@st.cache_data()
def plot_aggregates(df : pd.core.frame.DataFrame, main_col : str, annotate = False):

    sns.set_style('whitegrid')
    
    try:

        df_c = df[df['Continent'] != 'Unknown']
        df_c = df_c.groupby(main_col)[['Likes', 'Visits', 'Subscribers']].mean()


        
        fig, axes = plt.subplots(1, 3, figsize = (7, 7))
        
        for col, ax in zip(df_c.columns, axes.flat):
            sns.barplot(data = df_c, y = col, x = main_col, ax = ax, palette = 'Accent')
            ax.set_xlabel('')
            if ax == axes[0]:
                ax.set_ylabel('Average \n{} per\n Video'.format(col), rotation = 0, ha = 'right', va = 'center')
            elif ax == axes[1]:
                ax.set_ylabel('Average \n{} per\n Video'.format(col), rotation = 0, ha = 'right', va = 'center')
            elif ax == axes[2]:
                ax.set_ylabel('Average \n{} \nper Channel'.format(col), rotation = 0, ha = 'right', va = 'center')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)


            if annotate:
                for p in ax.patches:
                    height = p.get_height()
                    ax.annotate(f'{height:,.0f}', (p.get_x() + p.get_width() / 2, height),
                                ha='center', va='bottom', fontsize=8)  

            
        st.pyplot(plt.gcf())

        
        return fig, axes
    except Exception as e:
        return e
    
@st.cache_data
def make_pairplot():
    sns.set_style('white')
    fig2 = sns.pairplot(data[['Subscribers', 'Likes', 'Visits', 'Cluster']], hue='Cluster', palette='coolwarm')
    st.pyplot(plt.gcf())




# Model Evaluation functions
    
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


#function to retrieve youtube channel info
@st.cache_data
def extract_channel_info(url: str, category: str):
    """
    Extracts information about a YouTube channel, including subscriber count, country, continent, and video statistics.

    Parameters:
    - url (str): The URL of the YouTube channel.
    - category (str): The category or genre of the YouTube channel.

    Returns:
    - pd.DataFrame: A DataFrame containing information about the YouTube channel.

    This function queries the YouTube API to retrieve details about the specified channel, such as the channel's
    username, number of subscribers, country, continent, and average visits and likes for the latest 50 videos.
    The data is presented in a DataFrame for further analysis, and additional information is displayed using
    Streamlit for a user-friendly interface.

    Note: The API key should be stored in a file named 'api_key_lu.txt' in the same directory as this script.

    Example:
    >>> df = extract_channel_info("https://www.youtube.com/channel/UCxyz123", "Technology")
    """

    try:
        api_key = st.secrets['yt_api_key']['api_key']
    
        api_service_name = 'youtube'
        api_version = 'v3'

        url = url.strip('"') #In case of quotation marks
        youtube = build(
            api_service_name, api_version, developerKey = api_key
        )

        channel_id = url.split('/')[-1]  #the part of a channel's URL after the last '/' is the channel_id
        request = youtube.channels().list(
            part = 'snippet, contentDetails, statistics', 
            id = str(channel_id)
        )
        response = request.execute()


        for item in response['items']:
            yt_name = item['snippet']['title']
            yt_thumbnail_url = item['snippet']['thumbnails']['high']['url'] #to get the channel's thumbnail picture
            country_name = country_abbreviations.get(item['snippet']['country'], 'Unknown Country')

            data = {
                'Username': item['snippet']['title'],
                'Subscribers': item['statistics']['subscriberCount'],
                'Categories': category,
                'Country': country_name, 
                'Continent': continent_mapping.get(country_name, 'Unknown Continent')
            }


        # Since the YouTube API for channels can't retrieve video info, we need to make a separate query to get the averga visits and Likes for our channel

        referrer = "https://youtube-analytics-app-zahemen9900.streamlit.app/" # Replace this with your own domain name

        # Define the request URL and the headers

        vid_request_url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50"
        headers = {"Referer": referrer}

        # Make the GET request and print the response
        response_ = requests.get(vid_request_url, headers = headers)
        print('Status code is {}'.format(response_.status_code))

        vid_data = response_.json()

        video_titles, video_ids = [], []   #instantiate empty arrays to collect the video titles and ids

        # Loop through the items list
        for item in vid_data["items"]:
            # Get the video ID and title from the snippet dictionary
            video_id = item["id"]["videoId"]
            video_title = item["snippet"]["title"]

            # Append a tuple of video ID and title to the videos list
            video_titles.append(video_title)
            video_ids.append(video_id)


        request2 = youtube.videos().list(
            part = 'statistics',
            id = ','.join(video_ids) # A comma-separated list of video IDs
        )
        response2 = request2.execute()

        n_visits, n_likes = 0, 0
        # Sum up the like counts for the videos in the current page
        for item in response2['items']:
            n_visits += int(item['statistics']['viewCount'])
            n_likes += int(item['statistics']['likeCount'])

        n_visits /= 50
        n_likes /= 50

        data.update({
            'Visits': n_visits,
            'Likes': n_likes
        })

        yt_channel_df  = pd.DataFrame(data, index = [0])
        yt_channel_df.reindex(['Username', 'Subscribers', 'Category', 'Country', 'Continent', 'Visits', 'Likes']) #make sure the columns are arranged properly
        yt_channel_df = yt_channel_df.astype({
                                  'Username': 'object',
                                  'Subscribers': 'int64',
                                  'Categories': 'object',
                                  'Country': 'object',
                                  'Continent': 'object',
                                  'Visits': 'int64',
                                  'Likes': 'int64'
                              })

        st.write(f"""
                 ##### Hey _**{yt_name}**_, 
                 glad to have you here!
                 """)


        channel_thumbnail = st.image(yt_thumbnail_url, caption = yt_name)

        with st.expander('**Expand to see all your channel info**'):
            st.write('##### _**`channel_info`**_')
            st.write(response)

        st.write("##### Here's a summary of the relevant info:")
        st.write(yt_channel_df)

        with st.expander("_**Expand to see Videos we used**_"):
            formatted_titles = ' '.join([f'<li><b>{title}</b></li>' for title in video_titles])
            st.markdown(
                f"""
                ##### Here's a list of your latest 50 videos:
                ---
                <ul>
                {formatted_titles}
                </ul>
                """, unsafe_allow_html = True)

        return yt_channel_df
    


    except FileNotFoundError:
        st.write('Error: API key file not found in current directory')
    except Exception as e:
        st.write(f'An error occured: {e}')



        
def score_model(data: pd.core.frame.DataFrame, model, model_params: dict, scaled = False, encoded = False):
    """
    Trains and evaluates a machine learning model using the provided data.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the input features and target variable.
    - model (estimator): The machine learning model to be trained and evaluated.
    - model_params (dict): The hyperparameter grid for model tuning using GridSearchCV.
    - scaled (bool): If False, scales numeric features using StandardScaler (default is False).
    - encoded (bool): If False, encodes categorical features using LabelEncoder (default is False).

    Returns:
    - model_valid (estimator): The trained machine learning model with the best hyperparameters.
    """
    try:
    
        X = data.drop('Username', axis = 1).copy()
        
        y = X.pop('Cluster')
    
        if not scaled:
            scaler = StandardScaler()
            X[X.select_dtypes(include = ['number']).columns] = scaler.fit_transform(X[X.select_dtypes(include = ['number']).columns])
    
        if not encoded:
            for col in X.select_dtypes(include = ['object']).columns:
                label_encoder = LabelEncoder()
                X[col] = label_encoder.fit_transform(X[col])
    
    
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 5)
    
    
        grid_search = GridSearchCV(estimator = model, param_grid = model_params,
                                  scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 1)
    
        grid_search.fit(X_train, y_train)
        col1, col2 = st.columns([1, 1])

        col1.metric(label = 'Best Cross Validation Score', value = "{:.3f}".format(grid_search.best_score_))

        model_valid = grid_search.best_estimator_
        model_valid.fit(X_train, y_train)
    
        y_preds = model_valid.predict(X_valid)
    
        col2.metric(label = 'Accuracy Score for final model', value = ' {:.3f}'.format(grid_search.score(X_valid, y_valid)))

        st.write("**The best parameters are:**")
        st.write(grid_search.best_params_)
    
        return model_valid

    except Exception as e:
        st.write(f'An error occured: {e}')


#@st.cache_data()
def make_predictions(df: pd.core.frame.DataFrame, model, yt_channel_df, scaled = False, encoded = False):

    """
    Generates predictions for a YouTube channel using a trained machine learning model.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the input features and target variable used for model training.
    - model (estimator): The trained machine learning model used for making predictions.
    - scaled (bool): If True, scales numeric features using StandardScaler (default is False).
    - encoded (bool): If True, encodes categorical features using LabelEncoder (default is False).
    - **yt_channel_kwargs: Keyword arguments representing the YouTube channel Column Names. Must correspond to the Actual YouTube DataFrame Columns used for Analysis

    Returns:
    - str : A message indicating the predicted cluster for the given YouTube channel.
    """

    try:
        yt_channel_df.rename_axis('Rank', inplace = True)

        if all([col in df.columns for col in yt_channel_df.columns]):
            data = pd.concat([df, yt_channel_df], axis = 0, ignore_index = True)
        else:
            raise ValueError('Error: check that Input arguments match original dataframe columns.')
    
        if not scaled:
            scaler = StandardScaler()
            data[data.select_dtypes(include = ['number']).columns] = scaler.fit_transform(data[data.select_dtypes(include = ['number']).columns])
    
        if not encoded:
            for col in data.select_dtypes(include = ['object']).columns:
                label_encoder = LabelEncoder()
                data[col] = label_encoder.fit_transform(data[col])
    
        prediction = model.predict(data.tail(1).drop(['Username', 'Cluster'], axis = 1))

    
        st.write('Your predicted cluster : **{}**'.format(prediction[-1]))

        return prediction[-1]

    except Exception as e:
        st.write(f'An error occured: {e}')

#st.cache_data()
def generate_recommendations(df: pd.core.frame.DataFrame, yt_channel_df, model, scaled=False, encoded=False):

    result = make_predictions(df, model, yt_channel_df, scaled=False, encoded=False)

    cluster_descriptions = {
            1:  """\
                ### 🚀 **Rising Stars Category**

                Congratulations! Your channel belongs to the **Rising Stars** category. This means you're on the fast track to YouTube fame. You've gained millions of subscribers in a short period by creating unique and engaging content that appeals to a large audience. To keep up the momentum and reach the next level, here are some personalized recommendations for you:

                #### Some Popular Names:
                - **Dream:** This Minecraft gamer skyrocketed to fame in 2020 with his innovative and thrilling videos. He is known for his speedruns, manhunts, and collaborations with other popular YouTubers.
                - **Corpse Husband:** This mysterious and deep-voiced narrator started his channel in 2015, but gained massive popularity in 2020 with his horror stories and Among Us gameplay. He is also a singer and songwriter, and has collaborated with celebrities like Machine Gun Kelly.
                - **Emma Chamberlain:** This lifestyle vlogger and influencer rose to prominence in 2018 with her relatable and humorous videos. She has since branched out into podcasting, fashion, and coffee. She was named the "Most Popular YouTuber" by The New York Times in 2019.

                #### Characteristics:
                - Gained millions of subscribers in a short period
                - Created unique and engaging content
                - Appeals to a large audience

                #### Personalized Recommendations:
                - **Content Innovation:** You have a knack for creating **unique and engaging content** that sets you apart from the crowd. Keep experimenting with new ideas and formats, and don't be afraid to try something different. Find out what makes your content special and amplify those elements. For example, you can use analytics to see which videos perform best, and get feedback from your fans to see what they like and want more of.
                - **Social Media Boost:** You can leverage social media platforms to **promote your videos** and grow your fanbase. Engage with your audience on various platforms, such as Instagram, Twitter, TikTok, and Discord. Share behind-the-scenes content, teasers, polls, and giveaways. Interact with your fans by replying to their comments, messages, and tweets. This will help you increase your visibility, loyalty, and reach.
                - **Collaboration Power:** You can collaborate with other creators in your niche to expand your audience and learn from each other. Cross-promotion can lead to rapid growth and exposure. You can also join or create a YouTube group or network with other rising stars, such as the Dream SMP, the Vlog Squad, or the Sister Squad. This will help you build relationships, create more content, and have more fun.
                - **Consistent Effort:** You have the potential to become the next big thing on YouTube, so don't give up on your dreams. Stay consistent and passionate about your content, and your hard work will pay off. 🌟 You can also set goals and track your progress, such as reaching a certain number of views, subscribers, or revenue. Celebrate your achievements and milestones, and reward yourself for your efforts. You can also use tools and resources to help you grow your channel, such as [YouTube Creator Academy](https://www.youtube.com/creators/), [YouTube Analytics](https://studio.youtube.com/?csr=analytics), and [Biteable](https://biteable.com/).
                """
            ,

        2: """\
            ### 📉 **Ground Zero Category**

            Your channel is in the **Ground Zero** category, which means you're probably struggling to get visits, likes, and subscribers. This category is very crowded and competitive, and it's hard to stand out from the rest.

            #### Examples:
            Some Youtuber making it big time in this category are:
            - **The Dodo:** This channel features heartwarming stories of animals and their rescuers. It has over 11 million subscribers and billions of views. 
            - **Tasty:** This channel showcases easy and delicious recipes for all occasions. It has over 21 million subscribers and is one of the most popular food channels on YouTube.
            - **5-Minute Crafts:** This channel offers quick and simple DIY projects, hacks, and tips. It has over 74 million subscribers and is one of the most viewed channels on YouTube.

            #### Characteristics:
            - A lot of videos but low engagement rates
            - Relying on quantity over quality
            - Producing generic or clickbait content

            #### Personalized Recommendations:
            - **Content Strategy:** You need to rethink your **content strategy** and focus on quality over quantity. Instead of uploading a lot of videos that don't get much attention, try to create fewer but better videos that can attract and retain your viewers. Think about what value you can offer to your audience, and what problems you can solve for them. For example, you can use [YouTube Creator Academy](https://www.youtube.com/creators/) to learn how to plan, produce, and optimize your videos.
            - **Audience Targeting:** You need to target a **specific audience** that can relate to your content and engage with it. Instead of trying to appeal to everyone, try to find your niche and your ideal viewer persona. Think about who they are, what they like, what they need, and how you can reach them. For example, you can use [YouTube Analytics](https://studio.youtube.com/?csr=analytics) to understand your audience's demographics, interests, and behavior.
            - **Inspiration Analysis:** You need to analyze successful channels in your niche and get inspiration from them. Instead of copying or competing with them, try to learn from them and see what makes them popular and unique. Think about how you can differentiate yourself and offer something new or better. For example, you can use [Biteable](https://biteable.com/) to compare your channel with other channels and see how you can improve your performance.
            - **Analytics Tools:** You need to use **analytics tools** to measure and improve your channel's performance. Instead of relying on intuition or guesswork, try to use data and insights to guide your decisions and actions. Think about what goals you want to achieve, what metrics you want to track, and what actions you want to take. For example, you can use [Google Analytics](https://studio.youtube.com/?csr=analytics) to monitor and analyze your channel's traffic, conversions, and revenue.
            """
        ,
        3: """\
            ### 🛡️ **Subscribers' Haven Category**

            You belong to the **Subscribers' Haven** category, which means you have a large and loyal fan base that loves your content. However, you also face some challenges in terms of engagement and growth. Here are some tips to help you overcome them and take your channel to the next level.

            #### Examples:
            Some of the most successful YouTube channels in this category are:

            - **PewDiePie**: The king of YouTube, with over 100 million subscribers. He is known for his gaming videos, memes, and commentary.
            - **Mr Beast**: The philanthropist of YouTube, with over hundreds of millions of subscribers. He is known for his extravagant challenges, giveaways, and stunts.

            #### Characteristics:
            The main features of this category are:

            - **Loyal fan bases**: You have a dedicated audience that watches your videos regularly and supports you through various means.
            - **High retention rates**: Your viewers tend to watch your videos for a long time, indicating that they are interested and engaged in your content.
            - **Less frequent posting**: You upload videos less often than other categories, which may affect your visibility and reach.

            #### Recommendations:
            To optimize your performance in this category, you should:

            - **Build a stronger connection with your audience**: You already have a loyal fan base, but you can always make them feel more appreciated and involved. For example, you can interact with them more on social media, respond to their comments, ask for their feedback, or feature them in your videos.
            - **Encourage likes, comments, and shares**: These are the main indicators of engagement on YouTube, and they can help you boost your ranking and exposure. You can ask your viewers to like, comment, and share your videos at the beginning or end of your videos, or use incentives such as giveaways, shoutouts, or polls.
            - **Diversify your content while maintaining uniqueness**: You have a distinctive style and niche that sets you apart from other channels, but you can also explore new topics, genres, or formats that may appeal to your existing or potential viewers. For example, you can collaborate with other creators, try new trends, or experiment with different types of videos such as live streams, podcasts, or shorts.
            - **Keep your supporters entertained and satisfied**: You have the advantage of having a solid foundation of supporters, but you also have to meet their expectations and keep them interested in your content. You can do this by maintaining a consistent quality and frequency of your videos, updating them on your plans and projects, or surprising them with something special or unexpected. 🤝
            """
        ,
        4: """\
                ### ⚖️ **Balancing Act Category**

            You are in the **Balancing Act** category, which means you have a moderate but stable performance on YouTube. You have a decent number of visits, likes, and subscribers, and you create a variety of content that appeals to different audiences. However, you also face some challenges in terms of differentiation and growth. Here are some tips to help you stand out and reach your full potential.

            #### Examples:
            Some of the most popular YouTube channels in this category are:

            - **Katy Perry**: The pop star of YouTube, with over 40 million subscribers. She is known for her music videos, behind-the-scenes, and collaborations with other celebrities.
            - **The Ellen Show**: The talk show of YouTube, with over 38 million subscribers. She is known for her interviews, games, and pranks with famous guests and fans.

            #### Characteristics:
            The main features of this category are:

            - **Decent engagement rates**: You have a fair amount of likes, comments, and shares on your videos, indicating that your viewers are interested and engaged in your content.
            - **Creating a variety of content**: You produce different types of videos, such as entertainment, education, or lifestyle, that cater to different tastes and preferences.
            - **No clear niche or identity**: You do not have a specific focus or theme for your channel, which may make it harder for you to attract and retain a loyal fan base.

            #### Recommendations:
            To optimize your performance in this category, you should:

            - **Maintain a balance in your content**: You have the advantage of being versatile and flexible in your content creation, but you also have to be careful not to lose your identity or direction. You should balance your content between what you are passionate about and what your audience wants to see, and avoid spreading yourself too thin or jumping on every trend.
            - **Explore collaborations with creators in adjacent niches**: You can expand your audience and exposure by collaborating with other creators who have similar or complementary content to yours. For example, you can join forces with other musicians, comedians, or influencers, and create videos that showcase your talents and personalities.
            - **Be consistent in your content delivery**: You can increase your retention and growth rates by uploading videos regularly and consistently. You should establish a schedule and stick to it, and inform your viewers about your plans and updates. You can also use tools such as [YouTube Analytics](https://studio.youtube.com/?csr=analytics) or [Biteable](https://biteable.com/) to track your performance and optimize your content strategy.
            - **Optimize your SEO to improve your visibility and discoverability**: You can boost your ranking and reach on YouTube by using effective keywords, titles, descriptions, tags, and thumbnails for your videos. You should also use catchy and relevant hashtags, and encourage your viewers to subscribe and turn on notifications. You can also learn more about SEO best practices from YouTube Creator Academy or other online resources.
            - **Keep working hard and smart**: You have the potential to reach higher levels of success on YouTube, but you also have to work hard and smart to achieve your goals. You should always strive to improve your content quality and creativity, and keep learning from your feedback and analytics. You should also celebrate your achievements and milestones, and appreciate your supporters. 🚶‍♂️
            """
        ,
        5: """\
            ### 👥 **Engaging Echoes Category**

            You are in the **Engaging Echoes** category, which means you have a high-performance channel that attracts millions of views and likes. You create catchy or trendy content that resonates with a wide audience. However, you also have a low subscriber count compared to other channels, which means you have a challenge in retaining your viewers and building a loyal fan base. Here are some tips to help you turn your viewers into subscribers and grow your community.

            #### Examples:
            Some of the most viral YouTube channels in this category are:

            - **Techno Gamerz**: The gaming sensation of YouTube, with over 20 million subscribers. He is known for his gameplay videos, live streams, and challenges with other gamers.
            - **Kimberly Loaiza**: The queen of YouTube in Latin America, with over 30 million subscribers. She is known for her music videos, vlogs, and collaborations with other influencers.

            #### Characteristics:
            The main features of this category are:

            - **Millions of views and likes**: You have a huge reach and impact on YouTube, with your videos getting millions of views and likes in a short time. You have a knack for creating viral content that appeals to a mass audience.
            - **Catchy or trendy content**: You produce content that is relevant, timely, or entertaining, such as music, comedy, or news. You follow the latest trends and topics, and use effective strategies to capture attention and engagement.
            - **Not as many subscribers as other channels**: You have a lower subscriber count than other channels with similar or lower views and likes. This may indicate that your viewers are not as loyal or committed to your channel, and that they may watch your videos only once or occasionally.

            #### Recommendations:
            To optimize your performance in this category, you should:

            - **Work on strategies to convert viewers into subscribers**: You have the opportunity to grow your subscriber base by converting your viewers into subscribers. You can do this by using clear and compelling calls to action, such as asking your viewers to subscribe and turn on notifications at the beginning or end of your videos, or using pop-ups, cards, or end screens to remind them. You can also use tools such as YouTube Analytics to track your conversion rate and identify areas for improvement.
            - **Consider creating series or themed content to encourage consistent viewership**: You can increase your retention and loyalty rates by creating content that is consistent and coherent, such as series or themed content. For example, you can create a series of videos on a specific topic, genre, or format, and release them on a regular schedule. You can also create themed content based on seasons, events, or occasions, and use catchy titles and thumbnails to generate interest and anticipation. This way, you can keep your viewers hooked and coming back for more.
            - **Engage with your audience through comments and community posts to foster a loyal community**: You can strengthen your relationship with your audience by engaging with them through comments and community posts. You can respond to their comments, ask for their feedback, opinions, or suggestions, or create polls or quizzes to interact with them. You can also use community posts to update them on your plans, projects, or personal life, or share behind-the-scenes, sneak peeks, or teasers of your upcoming videos. This way, you can make your viewers feel valued and involved, and build a loyal community around your channel.
            - **Offer incentives such as giveaways, shoutouts, or merch to reward your fans**: You can motivate and reward your fans by offering them incentives such as giveaways, shoutouts, or merch. You can organize giveaways of products, services, or experiences that are related to your content or niche, and ask your viewers to subscribe, like, comment, or share your videos to enter. You can also give shoutouts to your fans who support you, or feature them in your videos. You can also create and sell merch such as t-shirts, hats, or mugs that represent your brand or personality, and promote them in your videos or social media. This way, you can show your appreciation and gratitude to your fans, and make them feel special and proud. 💬
            """
    }

    return cluster_descriptions.get(result, f"Oops, no specific information available for your cluster 🫤\n Make sure your channel's statistics like Country, etc. are properly recorded in your channel's info")






def main():
    global data, channel_name  # Make 'data' and 'channel_name' accessible globally

    #creating the side-bar

    st.set_page_config(initial_sidebar_state="expanded")
    with st.sidebar:
        st.markdown(
            """
            <style>
            h1 {
                font-family: Arial, Helvetica, sans-serif;
                font-size: 24px;
                font-weight: bold;
            }
            ul {
                list-style: none;
                padding: 0;
            }
            li {
                margin: 10px;
            }
            a {
                text-decoration: none;
            }
            a:link {
                color: brown;
            }
            a:hover {
                color: green;
            }
            a:visited {
                color: brown;
            }
            </style>
            <h1 style = "color: gray;">Table of Contents</h1>
            <ul>
                <li><b>🔗<a href="#youtube-channel-tip-app">Intro </a></b></li>
                <li><b>🔗<a href="#some-summary-statistics-on-the-data-used-based-on-different-categories">Summary Statistics </a></b></li>
                <li><b>🔗<a href="#a-summary-plot-for-metric-correlations-and-distributions">Pairplot Summary And Explanations </a></b></li>
                <li><b>🔗<a href="#pewdiepie-mr-beast-category-3">Top YouTubers in Each Category </a></b></li>
                <li><b>🔗<a href="#b40e990">Results from Model Training </a></b></li>
                <li><b>🔗<a href="#enter-your-channel-link-below">Recommendations Section (by channel URL)</a></b></li>
                <li><b>🔗<a href="#or-enter-your-details-manually-below">Recommendations Section(manual) </a></b></li>
                <li><b>🔗<a href="#more-on-data-used">About Data </a></b></li>
            </ul>
            """,
            unsafe_allow_html=True
        )


    data = get_csv_from_url()

    col1, col2 = st.columns([.7, .3])

    col1.markdown(
        """
        # <div style = "font-size: 75px; font-family: 'cooper black', 'sans-serif'; text-align: left;"><b>YouTube Channel Tip App</b></div>
        """, unsafe_allow_html = True
    )
    col2.markdown(
        """
        <div style = "text-align: top;">
        <img src = "https://cdn-icons-png.flaticon.com/256/1384/1384060.png" width = "300" height = "300" alt = "YouTube Logo"></div>
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

        <b><p style = "font-size: 20px;">Interested in taking your YouTube Channel Likes, Subscribers and Visits to the next level but not sure how?💭🤔</p><p style = "font-size: 20px;"> Get recommendations to boost your channel here!⚡📈</p></b>

        <p></p><p></p>
        <p style = "font-size: 20px;">Our analytics couldn't have been possible without curating some data from the Youtube API, and we wanted to show you our findings on some <b style = "color: brown;">Youtube Channel Analytics</b> first</p>

        <p style = "font-size: 22px;">Let's get started!✅</p>

        <p></p><p></p>
        """,
        unsafe_allow_html=True
    )


    st.write(

    )

    st.markdown(
        """
        ## <h3 style = "font-size: 45px;"><b>Some Summary Statistics on the data used based on different Categories</b></h3>
        <p></p><p></p><p></p>
        """,
        unsafe_allow_html=True
    )

    pie_plot()


    st.write(
        """
        #### Metric Performances across Continents:

        """
    )

    agg_plots = plot_aggregates(data, 'Continent', annotate = True)


    st.write(
        """
        #### A Summary plot for Metric correlations and distributions 
        ---
        """
    )


    # Create a pairplot using Seaborn
    pplot = make_pairplot()

    st.markdown(
        """
        <div style="border-color: #d3d3d3; border-width: 4px; border-style: solid; border-radius: 10px; padding: 15px; margin: 10px; font-family: Arial, Helvetica, sans-serif; box-shadow: 5px 5px 10px grey;">
          <h3 style="color: gray; margin-bottom: 10px;"><b>A Detailed Exploration of Channel Categories 🔍</b></h3>
          
          <h5 style="color: blue;"><b>Category 1: Rising Stars💫</b></h5> 
          <p>In this category, channels emerge with fewer visits, likes, and subscribers, steadily climbing the ranks of Top YouTubers. They embody the aspiring talents, on the verge of breaking into the mainstream.</p>
          
          <h5 style="color: #87ceeb;"><b>Category 2: Ground Zero📉</b></h5> 
          <p>This category represents the YouTubers on the lowest end of the spectrum. Among the top YouTubers, they fall behind the most in all the respective metrics, and are also very populous.</p>
          
          <h5 style="color: gray;"><b>Category 3: Subscribers' Haven✨</b></h5> 
          <p>This category hosts channels with a substantial subscriber base but modest likes and visits. Recognized for their high retention rates, they craft popular content, albeit at a less frequent pace, resulting in a distinct engagement pattern.</p>
          
          <h5 style="color: rgba(255, 218, 190, 1);"><b>Category 4: Balancing Act 🦾</b></h5> 
          <p>Moderate in visits, likes, and subscribers, these channels strike a balance on the lower spectrum, outshining <b style="color: #87ceeb;">Category 2</b> in overall metrics. They hold a middle ground, contributing to the diverse YouTube landscape.</p>
          
          <h5 style="color: #900C3F;"><b>Category 5: Engaging Echoes🔊</b></h5> 
          
          <p style = "color:default;">Channels in this category boast the highest likes and visits, yet maintain a humble subscriber count. They epitomize high engagement but wrestle with retention rates, creating a vibrant but fleeting viewership.</p>
        </div>

        <p></p>

        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size: 15px;"><p>Eager to see your recommendations? get them <a href = "https://youtube-analytics-app-zahemen9900.streamlit.app/~/+/#enter-your-channel-metrics" target = "_self"><b>Here</b></a>!</p>
        <p>But if you're loving the narrative, click the button below to see some other really cool insights we uncovered 👇</p></div>

        <p></p><p></p>
        """,
        unsafe_allow_html=True
    )


    with st.expander('**Click to see more Continent Info!🌍**'):
        st.write(
            """
            ##  How do content preferences vary across continents?:
            """
        )

        continent_prefs = preferences_across_continents(data)

        st.write(
            """
            Want to see all the details of the analysis? Check it out here in my **[Jupyter Notebook](https://github.com/zahemen9900/YouTube-Analytics-App/blob/main/YouTube_Data_EDA.ipynb)** 📓📊
            """
        )

    with st.expander('**Click to see the Top YouTubers in each Category!🎥**'):

        st.markdown(
        """
        <style>
            .channel-name {
                color: gray; 
            }
            .rounded-images {
                border-radius: 15px;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.4);
                overflow: hidden;
                margin-bottom: 20px;
            }

        </style>

        <div>
            <h2 class="channel-name"><b>PewDiePie & Mr Beast (Category 3)</b></h2>
            <p>PewDiePie is a Swedish YouTuber who is known for his gaming videos, comedy sketches, and meme reviews. He is <b>the most-subscribed individual creator</b> on YouTube with over <b>110 million subscribers</b>. Mr Beast is an American YouTuber who is famous for his expensive stunts, philanthropic acts, and viral challenges. He has over <b>80 million subscribers</b> and is one of the highest-earning YouTubers in the world.</p>
            <div class="rounded-images">
                <img src="https://hips.hearstapps.com/hmg-prod/images/pewdiepie_gettyimages-501661286.jpg?resize=1200:*" alt="PewDiePie" width="333", height = "350">
                <img src="https://wallpapers.com/images/hd/mr-beast-bright-screen-background-he6y102ildr4ca8q.jpg" alt="Mr Beast" width="333.32", height = "350">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>The Ellen Show & Katy Perry (Category 4)</b></h2>
            <p>The Ellen Show is an American daytime television variety comedy talk show hosted by Ellen DeGeneres. It has been running for <b>19 seasons</b> and has won numerous awards, including 11 Daytime Emmys for Outstanding Talk Show Entertainment. Katy Perry is an American singer, songwriter, and television personality. She is one of the best-selling music artists of all time, with over <b>143 million records sold worldwide</b>. She has nine U.S. number one singles and has received various accolades, including five American Music Awards and a Brit Award</p>
            <div class="rounded-images">
                <img src="https://m.media-amazon.com/images/M/MV5BODA5ZDQyMzYtZWQwMy00MDQ1LWE2OGUtNGYyNTk0Y2NhZGM4XkEyXkFqcGdeQXVyMTkzODUwNzk@._V1_.jpg" alt="The Ellen Show" width="333" height = "450">
                <img src="https://m.media-amazon.com/images/M/MV5BMjE4MDI3NDI2Nl5BMl5BanBnXkFtZTcwNjE5OTQwOA@@._V1_.jpg" alt="Katy Perry" width="333.32" height = "450">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>Techno Gamers & Kimberly Loaiza (Category 5)</b></h2>
            <p>Techno Gamers is an Indian gaming YouTuber who creates videos of gameplays and live streams of <b>GTA 5</b>, <b>Minecraft</b>, <b>Call of Duty</b>, and more. He has <b>over 37 million subscribers</b> and is one of the most popular gamers in India. Kimberly Loaiza is a Mexican internet personality and singer who started her YouTube career in 2016. She is currently the seventh most-followed user on TikTok and has over <b>40 million subscribers</b> on YouTube. She also has a music career and has released several singles, such as <em><b>Enamorarme</b>, <b>Patán</b></em>, and <em><b>Kitty</b></em>.</p>
            <div class="rounded-images">
                <img src="https://img.gurugamer.com/resize/740x-/2021/04/02/youtuber-ujjwal-techno-gamerz-3aa0.jpg" alt="Techno Gamerz" width="333" height = "450">
                <img src="https://m.media-amazon.com/images/I/71G48FB73WL._AC_UF1000,1000_QL80_.jpg" alt="Kimberly Loaiza" width="333.32" height = "450">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>SSSniperWolf & JackSepticEye (Category 1)</b></h2>
            <p>SSSniperWolf is a British-American YouTuber who is known for her gaming and reaction videos. She has over <b>30 million subscribers</b> and is one of the most-watched female gamers on YouTube. JackSepticEye is an Irish YouTuber who is also known for his gaming and vlog videos. He has over <b>27 million subscribers</b> and is one of the most influential Irish online personalities. He has also appeared in the film Free Guy and released a biographical documentary called <b><em>How Did We Get Here?</em></b></p>
            <div class="rounded-images">
                <img src="https://ih1.redbubble.net/image.2189561281.9428/mwo,x1000,ipad_2_skin-pad,750x1000,f8f8f8.u1.jpg" alt="SSSniper Wolf" width="333" height = "400">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Jacksepticeye_by_Gage_Skidmore.jpg/1200px-Jacksepticeye_by_Gage_Skidmore.jpg" alt="JackSepticEye" width="333.32" height = "400">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>JessNoLimit & Daddy Yankee (Category 2)</b></h2>
            <p>JessNoLimit is an Indonesian gaming YouTuber and Instagram star who is known for his Mobile Legends gameplays. He has over <b>42 million subscribers</b> and is the <b>third most-subscribed YouTuber in Indonesia</b>. Daddy Yankee is a Puerto Rican rapper, singer, songwriter, and actor who is considered the <b><em>"King of Reggaeton"</em></b>. He has sold over <b>30 million records worldwide</b> and has won numerous awards, including five Latin Grammy Awards and two Billboard Music Awards. He is best known for his hit songs like <em><b>Gasolina</b>, <b>Despacito</b></em>, and <em><b>Con Calma</b></em>.</p>
            <div class="rounded-images">
                <img src="https://akcdn.detik.net.id/visual/2023/05/05/jess-no-limit-dan-sisca-kohl-2_43.png?w=650&q=90" alt="JessNoLimit" width="333" height = "300">
                <img src="https://people.com/thmb/eT6A-wncUzuDs-XV08qRSd_gSUk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(688x281:690x283)/Daddy-Yankee-Retirement-120623-a855484297944821ad14c8b98453b6a5.jpg" alt="Daddy Yankee" width="333.32" height = "300">
            </div>
        </div>

        """,
        unsafe_allow_html=True
    )
        st.write("_** Ranked based on Popularity_")


    st.write("Before you jump to your results, how about a quick sneak peek into the magic behind them? Trust us, it's worth it!")

    st.write(
        """
        ### **The Magic Behind the Scenes ✨**
        
        ###### _(How We Trained and Validated Our Model)_
        ---

        """
    )

    st.write(
        """
        You might be curious about how we're generating the awesome recommendations for you. Well, the secret is **a powerful machine learning model**! We used **`Random Forest Regressor`** from Scikit-Learn, and with some clever hyperparameter-tuning with **`GridSearchCV`**, we achieved some amazing results! 🙌
        Here are some highlights from our training process:
        """
    )

    #After finding optimal parameters, I reduced grid parameters to improve load times
    model_params = {
    'n_estimators': [100],       # Number of trees in the forest
    'max_features': [0.5],     # Number of features to consider at each split
    'min_samples_split': [2],      # Minimum number of samples required to split a node
    'bootstrap': [True]        # Whether to use bootstrap samples when building trees
    }


    rf_model = score_model(data = data, model = RandomForestClassifier(), model_params = model_params)

    st.write("Check out the data we used [here](https://github.com/zahemen9900/YouTube-Analytics-App/blob/main/YouTube%20Data%20EDA/yt_cluster_data.csv)")
  


    st.markdown(
        """
        <div style="padding: 20px;"><p style = "font-size: 15px;"><em>....And finally, the moment you've been waiting for; </em></p><p style = "font-size: 20px;"><b>Proceed below to check your recommendations!⏬</b></div>

        <p></p><p></p>
        """,
        unsafe_allow_html=True
    )



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

        selected_cat = st.radio("#### **Choose your channel category**", ("Animation", "Toys", "Movies", "Video Games", "Music and Dance",
                                    "News and Politics", "Fashion", "Animals and Pets", "Education",
                                    "DIY and Life Hacks", "Fitness", "ASMR", "Comedy", "Technology", "Automobiles"))

        submit_btn = st.form_submit_button('Submit', help = 'Submit to see your channel ')

        while submit_btn:
            if not (yt_url.startswith('https://www.youtube.com/channel/') or yt_url.startswith('www.youtube.com/channel/') or yt_url.startswith('https://youtube.com/channel/') or yt_url.startswith('youtube.com/channel/')):
                st.error('Please enter a valid channel URL. Also make sure your link does not contain quotation marks')
                break
            else:
                st.success('Url received!')
                with st.spinner('Retrieving channel info...'):
                    time.sleep(1.5)
                new_channel_df = extract_channel_info(yt_url, selected_cat)

                with st.spinner('Getting your recommendations...'):
                    time.sleep(1)

                recs = generate_recommendations(df = data, yt_channel_df= new_channel_df, model = rf_model)

                st.write(recs)
                break



    st.write("### Or enter your details manually below")

#    formbtn = st.button("Form")

    with st.expander("**Expand to enter your info manually**"):
        if "formbtn_state" not in st.session_state:
            st.session_state.formbtn_state = False

        if st.session_state.formbtn_state:
            st.session_state.formbtn_state = True

        with st.form(key="channel_form"):

            channel_name = st.text_input('What is your channel name?', 'eg. zahemen9900')
            st.write("Select the category of your videos:")
            selected_category = st.radio("Options", ("Animation", "Toys", "Movies", "Video Games", "Music and Dance",
                                            "News and Politics", "Fashion", "Animals and Pets", "Education",
                                            "DIY and Life Hacks", "Fitness", "ASMR", "Comedy", "Technology", "Automobiles"))

            selected_country = st.selectbox("Select your country", popular_countries)

            default_continent = continent_mapping.get(selected_country, 'Unknown')

            continents = data['Continent'].unique()
            continents = [continent for continent in continents]

            # Get the index of the default continent in the list of continents
            # If the default continent is not in the list, use 0 as the index
            default_index = continents.index(default_continent) if default_continent in continents else 0

            selected_continent = st.selectbox("Select your Continent", continents, index = default_index)


            n_visits = st.slider("Number of Visits", min_value=0, max_value=int(5e6), value=5000)
            n_likes = st.slider("Average Likes per video", min_value=0, max_value=int(0.5e6), value=2500)
            n_subs = st.slider("Number of Subscribers on your channel", min_value=int(10e3), max_value=int(5e6), value= int(50e3))

            submit_button = st.form_submit_button(label="Submit", help = 'Submit to see your recommendations!')




            if submit_button:
                st.success('Form submitted, Results are underway!')
                time.sleep(1)

                with st.spinner('Loading recomendations'):
                    time.sleep(1)

                args =  pd.DataFrame({'Username' : channel_name,
                        'Categories': selected_category,
                        'Subscribers': n_subs,
                        'Country': selected_country,
                        'Continent' : selected_continent,
                        'Visits' : n_visits,
                        'Likes' : n_likes
                        }, index = [0])

                personalized = generate_recommendations(df = data, model = rf_model, yt_channel_df=args)


                st.write(personalized)


    st.markdown(
        """
        <p></p>
        <div style="font-size: 15px;">
        <h5><b>More on Data Used</b></h5>

        <p>As an aspiring Top YouTuber, you deserve to compare yourself with the best of the best. That's why we used a curated dataset of the world's top 1000 YouTubers, to give you accurate assessments and useful tips to level up your game. We hope you enjoyed the recommendation as much as we enjoyed making this project 😊</p>

        <p>If you are curious about how we made this app possible, or want to explore more resources for the project, you can check out <b>Zahemen's GitHub</b> <a href = "https://github.com/zahemen9900/YouTube-Analytics-App">here</a>. You will find the source code, the data, and more.</p>
        <p>Thank you for your time and attention!</p></div>

        """,
        unsafe_allow_html=True
    )



if __name__ == '__main__':
    main()
