import time
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from warnings import simplefilter

simplefilter('ignore')


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

    fig, ax = plt.subplots(figsize=(12, 7))
    fig = plt.pie(plt_data.values, labels = plt_data.index,
                  autopct='%1.2f%%', shadow = True,
                  colors=sns.color_palette('coolwarm', n_colors=len(plt_data)), startangle=90)

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

                ax.pie(category_counts.values, labels=category_counts.index, autopct = '%1.1f%%', startangle = 90,
                        colors = sns.color_palette('Spectral', len(category_counts) + 1), shadow = True)
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

#@st.cache_data()
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
        st.write("**The best parameters are: `{}`**".format(grid_search.best_params_))
        col1.metric(label = 'Best Cross Validation Score', value = "{:.3f}".format(grid_search.best_score_))

    
        model_valid = grid_search.best_estimator_
        model_valid.fit(X_train, y_train)
    
        y_preds = model_valid.predict(X_valid)
    
        col2.metric(label = 'Accuracy Score for final model', value = ' {:.3f}'.format(grid_search.score(X_valid, y_valid)))
    
    
        return model_valid

    except Exception as e:
        return e


#@st.cache_data()
def make_predictions(df: pd.core.frame.DataFrame, model, scaled = False, encoded = False, **yt_channel_kwargs):

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
        channel_data = pd.DataFrame(yt_channel_kwargs, index = [0])
        channel_data.rename_axis('Rank', inplace = True)

        if all([col in df.columns for col in channel_data.columns]):
            data = pd.concat([df, channel_data], axis = 0, ignore_index = True)
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
        return e

#st.cache_data()
def generate_recommendations(df: pd.core.frame.DataFrame, model, scaled=False, encoded=False, **yt_channel_kwargs):

    result = make_predictions(df, model, scaled=False, encoded=False, **yt_channel_kwargs)

    cluster_descriptions = {
        1:  "🚀 Your channel falls into the category of **Rising Stars**. This category is characterized by channels that are emerging with fewer visits, likes, and subscribers. You are the aspiring talents, steadily climbing the ranks of Top YouTubers."
            "Some examples of channels in this category are **Dream**, **Corpse Husband**, and **Emma Chamberlain**. They have gained millions of subscribers in a short period of time by creating unique and engaging content that appeals to a large audience."
            "To further grow, focus on creating **unique and engaging content** that sets you apart. Leverage social media platforms to **promote your videos** and **collaborate with other creators** in your niche. You have the potential to become the next big thing on YouTube, so don't give up on your dreams. 🌟"
        ,
        2: "📉 You find yourself in the **Ground Zero** category. Among the top YouTubers, you fall behind the most in visits, likes, and subscribers. This category is very populous and represents YouTubers on the lowest end of the spectrum."
            "Some examples of channels in this category are **The Dodo**, **Tasty**, and **5-Minute Crafts**. They have a lot of videos but low engagement rates. They rely on quantity over quality and often produce generic or clickbait content that does not retain viewers."
            "To rise above, consider refining your **content strategy** and targeting a **specific audience**. Analyze successful channels in your niche and incorporate similar elements into your videos to attract more engagement. You can also use **analytics tools** to understand your audience's preferences and behavior. You have the opportunity to improve your performance and stand out from the crowd. 📈"
        ,
        3: "🛡️ Welcome to **Subscribers' Haven**! Your channel boasts a substantial subscriber base but has modest likes and visits. Recognized for high retention rates, you craft popular content, albeit at a less frequent pace, resulting in a distinct engagement pattern."
            "Some examples of channels in this category are **PewDiePie** and **Mr Beast**. They have loyal fan bases that follow their content regularly, but they do not post as often as other channels. They focus on quality over quantity and often create viral or philanthropic content that generates a lot of buzz."
            "To thrive, focus on building a **stronger connection** with your audience. Encourage likes, comments, and shares to increase overall engagement. Consider **diversifying your content** while maintaining the unique elements that resonate with your subscribers. You can also **experiment with different formats** such as live streams, podcasts, or shorts to reach new audiences. You have the advantage of having a solid foundation of supporters, so keep them entertained and satisfied. 🤝"
        ,
        4: "⚖️ Your channel belongs to the **Balancing Act** category. Moderate in visits, likes, and subscribers, you strike a balance on the lower spectrum, outshining Category 2 in overall metrics. You hold a middle ground, contributing to the diverse YouTube landscape."
            "Some examples of channels in this category are **Katy Perry** and **The Ellen Show**. They have decent engagement rates but not as high as Category 5. They create a variety of content that appeals to different audiences, but they do not have a clear niche or identity."
            "To enhance your impact, continue maintaining a **balance** in your content. Explore **collaborations** with creators in adjacent niches to expand your audience. **Consistency** in content delivery will contribute to steady growth over time. You can also **optimize your SEO** to improve your visibility and discoverability. You have the potential to reach higher levels of success, so keep working hard and smart. 🚶‍♂️"
        ,
        5: "👥 Congratulations on being in the **Engaging Echoes** category! Your channel boasts the highest likes and visits, yet maintains a humble subscriber count. You epitomize high engagement but wrestle with retention rates, creating a vibrant but fleeting viewership."
            "Some examples of channels in this category are **Techno Gamerz** and **Kimberly Loaiza**. They have millions of views and likes on their videos, but they do not have as many subscribers as other channels. They create catchy or trendy content that attracts a lot of attention, but they do not have a strong bond with their viewers."
            "To solidify your presence, work on strategies to **convert viewers into subscribers**. Consider creating **series or themed content** to encourage consistent viewership. Engage with your audience through **comments and community posts** to foster a loyal community. You can also **offer incentives** such as giveaways, shoutouts, or merch to reward your fans. You have the advantage of having a large reach, so make the most of it. 💬"
    }

    return cluster_descriptions.get(result, "Oops, no specific information available for your cluster. 🥲")




def main():
    global data, channel_name  # Make 'data' and 'channel_name' accessible globally

    data = get_csv_from_url()

    st.markdown(
        """
        # <div style="font-size: 75px; font-family: 'Cambria', 'sans-serif'; text-align: left; color: brown; border-style: solid; border-width: 5px; border-radius: 10px; border-color: gray; padding: 20px; box-shadow: 5px 5px 10px grey;"><b>YouTube Channel Recommendation App</b></div>

        <p></p><p></p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
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
        #### A Summary plot for Metric correlations and distributions :
        """
    )


    # Create a pairplot using Seaborn
    pplot = make_pairplot()

    st.markdown(
        """
        <div style="background-color: #f5f5f5; border-color: #d3d3d3; border-width: 4px; border-style: solid; border-radius: 10px; padding: 15px; margin: 10px; font-family: Arial, Helvetica, sans-serif;">
          <h3 style="color: #333; margin-bottom: 10px;"><b>A Detailed Exploration of Channel Categories 🔍</b></h3>
          
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
                <img src="https://hips.hearstapps.com/hmg-prod/images/pewdiepie_gettyimages-501661286.jpg?resize=1200:*" alt="PewDiePie" width="333.33", height = "350">
                <img src="https://wallpapers.com/images/hd/mr-beast-bright-screen-background-he6y102ildr4ca8q.jpg" alt="Mr Beast" width="333.33", height = "350">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>The Ellen Show & Katy Perry (Category 4)</b></h2>
            <p>The Ellen Show is an American daytime television variety comedy talk show hosted by Ellen DeGeneres. It has been running for <b>19 seasons</b> and has won numerous awards, including 11 Daytime Emmys for Outstanding Talk Show Entertainment. Katy Perry is an American singer, songwriter, and television personality. She is one of the best-selling music artists of all time, with over <b>143 million records sold worldwide</b>. She has nine U.S. number one singles and has received various accolades, including five American Music Awards and a Brit Award</p>
            <div class="rounded-images">
                <img src="https://m.media-amazon.com/images/M/MV5BODA5ZDQyMzYtZWQwMy00MDQ1LWE2OGUtNGYyNTk0Y2NhZGM4XkEyXkFqcGdeQXVyMTkzODUwNzk@._V1_.jpg" alt="The Ellen Show" width="333.33" height = "450">
                <img src="https://m.media-amazon.com/images/M/MV5BMjE4MDI3NDI2Nl5BMl5BanBnXkFtZTcwNjE5OTQwOA@@._V1_.jpg" alt="Katy Perry" width="333.33" height = "450">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>Techno Gamers & Kimberly Loaiza (Category 5)</b></h2>
            <p>Techno Gamers is an Indian gaming YouTuber who creates videos of gameplays and live streams of <b>GTA 5</b>, <b>Minecraft</b>, <b>Call of Duty</b>, and more. He has <b>over 37 million subscribers</b> and is one of the most popular gamers in India. Kimberly Loaiza is a Mexican internet personality and singer who started her YouTube career in 2016. She is currently the seventh most-followed user on TikTok and has over <b>40 million subscribers</b> on YouTube. She also has a music career and has released several singles, such as <em><b>Enamorarme</b>, <b>Patán</b></em>, and <em><b>Kitty</b></em>.</p>
            <div class="rounded-images">
                <img src="https://img.gurugamer.com/resize/740x-/2021/04/02/youtuber-ujjwal-techno-gamerz-3aa0.jpg" alt="Techno Gamerz" width="333.33" height = "450">
                <img src="https://m.media-amazon.com/images/I/71G48FB73WL._AC_UF1000,1000_QL80_.jpg" alt="Kimberly Loaiza" width="333.33" height = "450">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>SSSniperWolf & JackSepticEye (Category 1)</b></h2>
            <p>SSSniperWolf is a British-American YouTuber who is known for her gaming and reaction videos. She has over <b>30 million subscribers</b> and is one of the most-watched female gamers on YouTube. JackSepticEye is an Irish YouTuber who is also known for his gaming and vlog videos. He has over <b>27 million subscribers</b> and is one of the most influential Irish online personalities. He has also appeared in the film Free Guy and released a biographical documentary called <b><em>How Did We Get Here?</em></b></p>
            <div class="rounded-images">
                <img src="https://ih1.redbubble.net/image.2189561281.9428/mwo,x1000,ipad_2_skin-pad,750x1000,f8f8f8.u1.jpg" alt="SSSniper Wolf" width="333.33" height = "400">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Jacksepticeye_by_Gage_Skidmore.jpg/1200px-Jacksepticeye_by_Gage_Skidmore.jpg" alt="JackSepticEye" width="333.33" height = "400">
            </div>
            <p></p><p></p>
        </div>

        <div>
            <h2 class="channel-name"><b>JessNoLimit & Daddy Yankee (Category 2)</b></h2>
            <p>JessNoLimit is an Indonesian gaming YouTuber and Instagram star who is known for his Mobile Legends gameplays. He has over <b>42 million subscribers</b> and is the <b>third most-subscribed YouTuber in Indonesia</b>. Daddy Yankee is a Puerto Rican rapper, singer, songwriter, and actor who is considered the <b><em>"King of Reggaeton"</em></b>. He has sold over <b>30 million records worldwide</b> and has won numerous awards, including five Latin Grammy Awards and two Billboard Music Awards. He is best known for his hit songs like <em><b>Gasolina</b>, <b>Despacito</b></em>, and <em><b>Con Calma</b></em>.</p>
            <div class="rounded-images">
                <img src="https://akcdn.detik.net.id/visual/2023/05/05/jess-no-limit-dan-sisca-kohl-2_43.png?w=650&q=90" alt="JessNoLimit" width="333.33" height = "300">
                <img src="https://people.com/thmb/eT6A-wncUzuDs-XV08qRSd_gSUk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(688x281:690x283)/Daddy-Yankee-Retirement-120623-a855484297944821ad14c8b98453b6a5.jpg" alt="Daddy Yankee" width="333.33" height = "300">
            </div>
        </div>

        """,
        unsafe_allow_html=True
    )
        st.write("_** Ranked based on Popularity_")



    st.write(
        """
        ### **Results from Model Training & Validation**
        ---

        """
    )

    st.write(
        """
        In case you wondered, the model for making the recommendations is a **`Random Forest Regressor`** from Scikit-Learn's **`sklearn.ensemble`**, and thanks to some advanced hyperparameter-tuning with **`GridSearchCV`**, we were able to produce some pretty robust results! 💯
        Here are summary results from training:
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



    st.markdown(
        """
        <div style="font-size: 20px;"><p>....And finally, the moment you've been waiting for; </p><p>Proceed below to check your recommendations!⏬</div>

        <p></p><p></p>
        """,
        unsafe_allow_html=True
    )

    st.header("Enter your channel Metrics")

#    formbtn = st.button("Form")

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

        selected_continent = st.selectbox("Select your Continent", data['Continent'].unique())

        n_visits = st.slider("Number of Visits", min_value=0, max_value=int(10e6), value=5000)
        n_likes = st.slider("Average Likes per video", min_value=0, max_value=int(1e6), value=2500)
        n_subs = st.slider("Number of Subscribers on your channel", min_value=0, max_value=int(5e6), value= int(50e3))

        submit_button = st.form_submit_button(label="Submit", help = 'Submit to see your recommendations!')




        if submit_button:

            st.success('Form submitted, Results are underway!')

            time.sleep(1)


            personalized = generate_recommendations(df = data, model = rf_model,
                         Username = channel_name, Categories = selected_category, Subscribers = n_subs, Country = selected_country, Continent = selected_continent,
                          Visits = n_visits, Likes = n_likes)


            st.write(personalized)

    st.markdown(
        """
        <p></p><p></p></p><p></p><p></p>
        <div style="font-size: 15px;">
        <h5><b>More on Data Used</b></h5>

        <p>Because you want to be a Top YouTuber, it only makes sense to assess yourself with the Game Changers; The dataset used in this project is a curated set of the world's top 1000 YouTubers, to make assessments and help you move to the next level. We hope you loved the recommendation as much as we loved making this project 😊</p>

        <p>To view the data and source code, as well as other assets for the project, check them out on Zahemen's GitHub <a href = "https://github.com/zahemen9900/YouTube-Analytics-App">Here</a></p>
        <p>Thanks for your time!</p></div>

        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
