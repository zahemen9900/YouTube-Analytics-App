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



st.session_state['channel_name'] = ''

def get_csv_from_loc(loc = r"https://raw.githubusercontent.com/zahemen9900/YouTube-Analytics-App/main/YouTube%20Data%20EDA/yt_cluster_data.csv"):
    if 'yt_data' not in st.session_state:
        st.session_state['yt_data'] = pd.read_csv(loc)
    df = st.session_state['yt_data']

    return df

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


        st.session_state['channel_name'] = yt_name


        # Since the YouTube API for channels can't retrieve video info, we need to make a separate query to get the averga visits and Likes for our channel

        referrer = st.secrets['referrers']['referrer_site'] # Replace this with your own domain name

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





def load_objects(objects_path = ''):
    """
    Function to load the objects for inferencing.
    """
    PATH = os.getcwd()
    if len(folder_name) < 1:
        pipeline_path = os.path.join(PATH, 'yt_pipeline.joblib')
        cpm_scaler_path = os.path.join(PATH, 'cpm_scaler.joblib')
        model_path = os.path.join(PATH, 'yt_model.joblib')
        tokenizer_path = os.path.join(PATH, 'yt_tokenizer.joblib')
    else:
        pipeline_path = os.path.join(PATH, f'{folder_name}/yt_pipeline.joblib')
        cpm_scaler_path = os.path.join(PATH, f'{folder_name}/cpm_scaler.joblib')
        model_path = os.path.join(PATH, f'{folder_name}/yt_model.joblib')
        tokenizer_path = os.path.join(PATH, f'{folder_name}/yt_tokenizer.joblib')

    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
    else:
        pipeline = None
        print(f"{pipeline_path} does not exist.")

    if os.path.exists(cpm_scaler_path):
        cpm_scaler = joblib.load(cpm_scaler_path)
    else:
        cpm_scaler = None
        print(f"{cpm_scaler_path} does not exist.")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None
        print(f"{model_path} does not exist.")

    if os.path.exists(tokenizer_path):
        tokenizer = joblib.load(tokenizer_path)
    else:
        tokenizer = None
        print(f"{tokenizer_path} does not exist.")

    return pipeline, cpm_scaler, model, tokenizer

# # Usage example
# pipeline_l, cpm_scaler_l, model_l, tokenizer_l = load_objects()





#Definitions of Alpha, Beta and Gamma are defined in the notebook 'YouTube Recommendation Engine' in main branch
def predict_cluster(user_input: Union[dict, pd.DataFrame], model, pipeline, scaler, tokenizer, alpha = 0.3, beta = 0.5, gamma = 0.2):
    """
    Predict the cluster for the given user input.

    Parameters:
    user_input (Union[dict, pd.DataFrame]): The input data for prediction. Can be a dictionary or a pandas DataFrame.
    model: The trained model used for prediction.
    pipeline: The preprocessing pipeline.
    scaler: The scaler used for scaling the input data. Default from model build is an sklearn.preprocessing.MinMaxScaler object.
    tokenizer: The tokenizer used for text data.
    alpha (float): The weight for the first component. Default is 0.3.
    beta (float): The weight for the second component. Default is 0.5.
    gamma (float): The weight for the third component. Default is 0.2.

    Returns:
    Prediction result from the model.
    """
    if isinstance(user_input, pd.DataFrame):
        user_input_df = user_input
    else:
        user_input_df = pd.DataFrame([user_input])

    user_metrics = scaler.transform(user_input_df[['Subscribers', 'Visits', 'Likes']])
    user_metrics_df = pd.DataFrame(user_metrics, columns=['Subscribers_normalized', 'Visits_normalized', 'Likes_normalized'])

    user_input_df['CPM'] = alpha * user_metrics_df['Subscribers_normalized'] + \
                        beta * user_metrics_df['Visits_normalized'] + \
                        gamma * user_metrics_df['Likes_normalized']

    user_input_features = pipeline.transform(user_input_df.drop('Username', axis = 1))

    display(user_input_df.head())

    username_sequence = tokenizer.texts_to_sequences([user_input['Username']])
    user_padded_sequence = pad_sequences(username_sequence, maxlen = 3, padding = 'post', truncating = 'post')

    # Make predictions
    predictions = model.predict([user_padded_sequence, user_input_features])

    # Process predictions for classification
    predicted_classes = np.argmax(predictions, axis=1)

    return predicted_classes[0]




def generate_recommendations(user_input: dict, model_l, pipeline_l, cpm_scaler_l, tokenizer_l):
    """
    Generate personalized recommendations based on user input using a trained model and associated preprocessing objects.

    This function takes user input, checks if specific preprocessing and model objects exist in a designated folder,
    loads them accordingly, predicts the user's cluster using these objects, and provides recommendations based on the predicted cluster.

    Parameters:
    user_input (dict): A dictionary containing the user's input data.
    model_l: The trained model object used for predicting the user's cluster.
    pipeline_l: The preprocessing pipeline object.
    cpm_scaler_l: The scaler object used for scaling the input data.
    tokenizer_l: The tokenizer object used for processing text data.

    Returns:
    str: A string containing personalized recommendations based on the predicted cluster.

    Example:
    >>> user_input = {"views": 1000, "subscribers": 100, "engagement": 50}
    >>> recommendations = generate_recommendations(user_input, model, pipeline, scaler, tokenizer)
    >>> print(recommendations)
    """
    
    #check if the "objects" folder exists, else use default location for retrieval
    if os.path.exists(os.path.join(os.getcwd(), "objects")):
        pipeline_l, cpm_scaler_l, model_l, tokenizer_l = load_objects('objects')
    else:
        pipeline_l, cpm_scaler_l, model_l, tokenizer_l = load_objects()

    result = predict_cluster(user_input, model_l, pipeline_l, cpm_scaler_l, tokenizer_l)
    
    cluster_descriptions = {
        0: """\
            ### 📉 **Ground Zero Category**
    
            Your channel is in the **Ground Zero** category, which means you're probably struggling to get visits, likes, and subscribers. This category is very crowded and competitive, and it's hard to stand out from the rest.
    
            #### Examples:
            Some Youtubers making it big in this category are:
            - **The Dodo:** This channel features heartwarming stories of animals and their rescuers. It has over 11 million subscribers and billions of views.
            - **Tasty:** This channel showcases easy and delicious recipes for all occasions. It has over 21 million subscribers and is one of the most popular food channels on YouTube.
            - **5-Minute Crafts:** This channel offers quick and simple DIY projects, hacks, and tips. It has over 74 million subscribers and is one of the most viewed channels on YouTube.
    
            #### Characteristics:
            - A lot of videos but low engagement rates
            - Relying on quantity over quality
            - Producing generic or clickbait content
    
            #### Personalized Recommendations:
            - **Content Strategy:** Rethink your **content strategy** and focus on quality over quantity. Instead of uploading numerous videos that don't get much attention, try to create fewer but higher-quality videos that can attract and retain your viewers. Consider what value you can offer to your audience and what problems you can solve for them. Utilize resources like [YouTube Creator Academy](https://www.youtube.com/creators/) to plan, produce, and optimize your videos.
            - **Audience Targeting:** Target a **specific audience** that can relate to your content and engage with it. Instead of trying to appeal to everyone, find your niche and ideal viewer persona. Understand who they are, what they like, and how you can reach them. Use [YouTube Analytics](https://studio.youtube.com/?csr=analytics) to gain insights into your audience's demographics, interests, and behavior.
            - **Inspiration Analysis:** Analyze successful channels in your niche and get inspiration from them. Learn from them to see what makes them popular and unique. Think about how you can differentiate yourself and offer something new or better. Use tools like [Biteable](https://biteable.com/) to compare your channel with others and see how you can improve your performance.
            - **Analytics Tools:** Use **analytics tools** to measure and improve your channel's performance. Rely on data and insights rather than intuition or guesswork. Set goals, track metrics, and take informed actions. Use [Google Analytics](https://studio.youtube.com/?csr=analytics) to monitor and analyze your channel's traffic, conversions, and revenue.
        """,
        
        1: """\
            ### 🛡️ **Subscribers' Haven Category**
    
            You belong to the **Subscribers' Haven** category, which means you have a large and loyal fan base that loves your content. However, you also face challenges in terms of engagement and growth. Here are some tips to help you overcome them and take your channel to the next level.
    
            #### Examples:
            Some of the most successful YouTube channels in this category are:
            - **PewDiePie:** The king of YouTube, with over 100 million subscribers. Known for his gaming videos, memes, and commentary.
            - **Mr Beast:** The philanthropist of YouTube, with over hundreds of millions of subscribers. Known for his extravagant challenges, giveaways, and stunts.
    
            #### Characteristics:
            - **Loyal fan bases:** You have a dedicated audience that watches your videos regularly and supports you through various means.
            - **High retention rates:** Your viewers tend to watch your videos for a long time, indicating they are interested and engaged in your content.
            - **Less frequent posting:** You upload videos less often than other categories, which may affect your visibility and reach.
    
            #### Recommendations:
            - **Build a stronger connection with your audience:** Interact with them more on social media, respond to comments, ask for feedback, or feature them in your videos.
            - **Encourage likes, comments, and shares:** These are the main indicators of engagement on YouTube. Ask your viewers to like, comment, and share your videos, or use incentives like giveaways or shoutouts.
            - **Diversify your content while maintaining uniqueness:** Explore new topics, genres, or formats that may appeal to your existing or potential viewers. Collaborate with other creators, try new trends, or experiment with different video types.
            - **Keep your supporters entertained and satisfied:** Maintain a consistent quality and frequency of your videos, update your viewers on plans and projects, or surprise them with something special or unexpected. 🤝
        """,
        
        2: """\
            ### 👥 **Engaging Echoes Category**
    
            You are in the **Engaging Echoes** category, which means you have a high-performance channel that attracts millions of views and likes. You create catchy or trendy content that resonates with a wide audience. However, you also have a low subscriber count compared to other channels, indicating a challenge in retaining your viewers and building a loyal fan base. Here are some tips to help you turn your viewers into subscribers and grow your community.
    
            #### Examples:
            Some of the most viral YouTube channels in this category are:
            - **Techno Gamerz:** The gaming sensation of YouTube, with over 20 million subscribers. Known for gameplay videos, live streams, and challenges with other gamers.
            - **Kimberly Loaiza:** The queen of YouTube in Latin America, with over 30 million subscribers. Known for music videos, vlogs, and collaborations with other influencers.
    
            #### Characteristics:
            - **Millions of views and likes:** You have a huge reach and impact on YouTube, with your videos getting millions of views and likes in a short time.
            - **Catchy or trendy content:** You produce content that is relevant, timely, or entertaining, such as music, comedy, or news.
            - **Not as many subscribers as other channels:** You have a lower subscriber count than other channels with similar or lower views and likes.
    
            #### Recommendations:
            - **Work on strategies to convert viewers into subscribers:** Use clear and compelling calls to action, such as asking viewers to subscribe and turn on notifications, or using pop-ups, cards, or end screens.
            - **Consider creating series or themed content to encourage consistent viewership:** Create content that is consistent and coherent, such as series or themed content, and release them on a regular schedule.
            - **Engage with your audience through comments and community posts to foster a loyal community:** Respond to comments, ask for feedback, or create polls or quizzes to interact with your audience.
            - **Offer incentives such as giveaways, shoutouts, or merch to reward your fans:** Motivate and reward your fans by organizing giveaways, giving shoutouts, or creating and selling merch that represents your brand or personality. 💬
        """,
        
        3: """\
            ### 🚀 **Rising Stars Category**
    
            Congratulations! Your channel belongs to the **Rising Stars** category. This means you're on the fast track to YouTube fame. You've gained millions of subscribers in a short period by creating unique and engaging content that appeals to a large audience. To keep up the momentum and reach the next level, here are some personalized recommendations for you:
    
            #### Some Popular Names:
            - **Dream:** This Minecraft gamer skyrocketed to fame in 2020 with his innovative and thrilling videos. Known for his speedruns, manhunts, and collaborations with other popular YouTubers.
            - **Corpse Husband:** This mysterious and deep-voiced narrator gained massive popularity in 2020 with his horror stories and Among Us gameplay. He is also a singer and songwriter and has collaborated with celebrities like Machine Gun Kelly.
            - **Emma Chamberlain:** This lifestyle vlogger and influencer rose to prominence in 2018 with her relatable and humorous videos. She has since branched out into podcasting, fashion, and coffee. Named the "Most Popular YouTuber" by The New York Times in 2019.
    
            #### Characteristics:
            - Gained millions of subscribers in a short period
            - Created unique and engaging content
            - Appeals to a large audience
    
            #### Personalized Recommendations:
            - **Content Innovation:** Continue creating **unique and engaging content**. Experiment with new ideas and formats, and don't be afraid to try something different. Use analytics to see which videos perform best and get feedback from your fans.
            - **Social Media Boost:** Leverage social media platforms to **promote your videos** and grow your fanbase. Engage with your audience on various platforms, such as Instagram, Twitter, TikTok, and Discord.
            - **Collaboration Power:** Collaborate with other creators in your niche to expand your audience and learn from each other. Cross-promotion can lead to rapid growth and exposure.
            - **Consistent Effort:** Stay consistent and passionate about your content. Set goals and track your progress, such as reaching a certain number of views, subscribers, or revenue. Celebrate your achievements and reward yourself for your efforts. 🌟 Use resources like [YouTube Creator Academy](https://www.youtube.com/creators/), [YouTube Analytics](https://studio.youtube.com/?csr=analytics), and [Biteable](https://biteable.com/).
        """,
        
        4: """\
            ### ⚖️ **Balancing Act Category**
    
            You are in the **Balancing Act** category, which means you have a moderate but stable performance on YouTube. You have a decent number of visits, likes, and subscribers, and you create a variety of content that appeals to different audiences. However, you may also face challenges in terms of maintaining and growing your channel, as well as managing your time and resources.
    
            #### Examples:
            Some YouTubers who fit this category are:
            - **Casey Neistat:** This vlogger and filmmaker has over 12 million subscribers and creates inspiring and entertaining videos about his life and adventures.
            - **MKBHD:** This tech reviewer has over 19 million subscribers and produces high-quality and informative videos about the latest gadgets and innovations.
            - **Lilly Singh:** This comedian and actress has over 14 million subscribers and creates hilarious and relatable videos about her culture and experiences.
    
            #### Characteristics:
            - **Moderate but stable performance:** You have a consistent but not exceptional level of engagement on YouTube, with your videos getting a fair amount of views, likes, and comments.
            - **Variety of content:** You produce content that covers different topics, genres, or formats, such as vlogs, reviews, tutorials, or skits.
            - **Appealing to different audiences:** You attract viewers from various demographics, interests, and regions, who may have different preferences and expectations.
    
            #### Recommendations:
            - **Focus on your strengths and passions:** Identify what makes you unique and what you enjoy creating the most. Highlight your strengths and passions in your videos, and create content that reflects your personality and values.
            - **Plan your content calendar:** Organize your content creation process by planning ahead and setting realistic goals. Use a content calendar to schedule your uploads, promotions, and collaborations. Stick to a consistent and manageable routine that works for you and your audience.
            - **Balance your work and life:** Avoid burnout and stress by balancing your work and life. Take breaks, delegate tasks, or seek help from others. Prioritize your health and well-being, and make time for your hobbies, family, and friends.
            - **Seek feedback and improvement:** Learn from your audience and peers by seeking feedback and improvement. Read comments, surveys, or reviews, and respond to constructive criticism. Use analytics and insights to monitor your performance and identify areas for improvement. Use tools like [TubeBuddy](https://www.tubebuddy.com/) and [VidIQ](https://vidiq.com/) to optimize your channel and videos. ✨
        """
    }

    return cluster_descriptions.get(result, "Cluster description not found.\n Make sure the link is valid, and that your channel's statistics like Country, etc. are properly recorded in your channel's YouTube database.")





# Create a form for the user to enter their email address
def deliver_recommendations(email_form_key, recommendations):
  st.markdown("""<h4 style = "font-family: Calibri, sans-serif">Get your recommendations delivered straight into your mail! 📩<h4>""", unsafe_allow_html = True)
  with st.form(key= email_form_key):
    email_input = st.text_input('Enter your email address')
    submit_button = st.form_submit_button('Send email')


  if submit_button:
    try:
      # Validate the email input using a regular expression
      #email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
      #if not re.match(email_pattern, email_input):
      #  st.error('Invalid email address')
      #  submit_button = False

      # Create a MIMEText object for the email body
      recommendations = f"Hey there, {st.session_state['channel_name']}, here are your recommendations:\n\n\n" + recommendations.replace('#', '').replace('_', '').replace('*', '').replace('[', '').replace(']', '')
      msg = MIMEText(recommendations, "plain")

      app_email = st.secrets['emails']['app_email']
      password = st.secrets['remote_ps']['password']

      # Add the email headers
      msg['From'] = app_email
      msg['To'] = email_input
      msg['Subject'] = 'Your YouTube recommendations'

      # Send the email using Gmail SMTP server
      server = smtplib.SMTP('smtp.gmail.com', 587)
      server.starttls()
      server.login('yt.analytics.app.z@gmail.com', password)
      server.sendmail('yt.analytics.app.z@gmail.com', email_input, msg.as_string())
      server.quit()

      with st.spinner('Sending your recommendations...'):
        time.sleep(1)
      st.success('Email sent successfully!')
      st.write("Please check your spam folder if you don't see it in your inbox")

    except Exception as e:
      st.write(e)




#App Config & Other elements

def give_feedback():

  with st.form(key='feedback_form'):
    email_input = st.text_input('Your email address _(so we can reach out to you)_ or just your first name')
    recommendations = st.text_input('Please enter feedback or recommendations here')
    submit_button = st.form_submit_button('Submit', help = 'deliver feedback')


  while submit_button:
    try:
      if len(recommendations) < 5:
        st.error('Recommendation too short. Please enter a valid response.')
        submit_button = False
      # Create a MIMEText object for the email body

      seconds = time.time()
      localtime = time.localtime(seconds)
      time_sent = time.asctime(localtime)

      recommendations = f"{email_input} gave the following feedback on {time_sent}:\n\n" + recommendations

      msg = MIMEText(recommendations, "plain")

      # Add the email headers
      my_email = st.secrets['emails']['my_email']
      app_email = st.secrets['emails']['app_email']
      password = st.secrets['remote_ps']['password']

      msg['From'] = app_email
      msg['To'] = my_email
      msg['Subject'] = 'Feedback from {}'.format(email_input)

      # Send the email using Gmail SMTP server

      st.write()
      server = smtplib.SMTP('smtp.gmail.com', 587)
      server.starttls()

      server.login(app_email, password)
      server.sendmail(app_email, my_email, msg.as_string())
      server.quit()

      with st.spinner('Sending response...'):
        time.sleep(1)

      st.success('Response received. Thanks for your feedback!😊')
      break

    except Exception as e:
      st.write(e)
      break


