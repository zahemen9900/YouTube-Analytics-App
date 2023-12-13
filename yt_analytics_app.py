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
            ax.set_ylabel('Average \n{}'.format(col), rotation = 0, ha = 'right', va = 'center')
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
        

# Model Evaluation functions

#@st.cache_data()
def score_model(data, model, model_params, scaled = False, encoded = False):
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
def make_predictions(df, model, scaled = False, encoded = False, **yt_channel_kwargs):
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

    
        st.write('Your predicted cluster : {}'.format(prediction[-1]))

        return prediction[-1]

    except Exception as e:
        return e

#st.cache_data()
def generate_recommendations(df, model, scaled=False, encoded=False, **yt_channel_kwargs):

    result = make_predictions(df, model, scaled=False, encoded=False, **yt_channel_kwargs)

    cluster_descriptions = {
        1:  "🚀 Your channel falls into the category of Rising Stars. This category is characterized by channels that are emerging with fewer visits, likes, and subscribers. You are the aspiring talents, steadily climbing the ranks of Top YouTubers."
            "To further grow, focus on creating unique and engaging content that sets you apart. Leverage social media platforms to promote your videos and collaborate with other creators in your niche. 🌟"
        ,
        2: "📉 You find yourself in the Ground Zero category. Among the top YouTubers, you fall behind the most in visits, likes, and subscribers. This category is very populous and represents YouTubers on the lowest end of the spectrum."
            "To rise above, consider refining your content strategy and targeting a specific audience. Analyze successful channels in your niche and incorporate similar elements into your videos to attract more engagement. 📈"
        ,
        3: "🛡️ Welcome to Subscribers' Haven! Your channel boasts a substantial subscriber base but has modest likes and visits. Recognized for high retention rates, you craft popular content, albeit at a less frequent pace, resulting in a distinct engagement pattern."
            "To thrive, focus on building a stronger connection with your audience. Encourage likes, comments, and shares to increase overall engagement. Consider diversifying your content while maintaining the unique elements that resonate with your subscribers. 🤝"
        ,
        4: "⚖️ Your channel belongs to the Balancing Act category. Moderate in visits, likes, and subscribers, you strike a balance on the lower spectrum, outshining Category 2 in overall metrics. You hold a middle ground, contributing to the diverse YouTube landscape."
            "To enhance your impact, continue maintaining a balance in your content. Explore collaborations with creators in adjacent niches to expand your audience. Consistency in content delivery will contribute to steady growth over time. 🚶‍♂️"
        ,
        5: "👥 Congratulations on being in the Engaging Echoes category! Your channel boasts the highest likes and visits, yet maintains a humble subscriber count. You epitomize high engagement but wrestle with retention rates, creating a vibrant but fleeting viewership."
            "To solidify your presence, work on strategies to convert viewers into subscribers. Consider creating series or themed content to encourage consistent viewership. Engage with your audience through comments and community posts to foster a loyal community. 💬"
    }

    return cluster_descriptions.get(result, "Oops, no specific information available for your cluster. 🥲")




def main():
    global data, channel_name  # Make 'data' accessible in the 'main' function

    data = pd.read_csv(r"C:\Users\David Yeboah\Documents\yt_cluster_data.csv")

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

    plot_aggregates(data, 'Continent', annotate = True)


    st.write(
        """
        #### A Summary plot for Metric correlations and distributions :
        """
    )


    # Create a pairplot using Seaborn
   # plt.figure(figsize=(30, 30), dpi=150)
    fig2 = sns.pairplot(data[['Subscribers', 'Likes', 'Visits', 'Cluster']], hue='Cluster', palette='coolwarm')
    st.pyplot(plt.gcf())

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
          
          <h5 style="color: brown;"><b>Category 5: Engaging Echoes🔊</b></h5> 
          
          <p style = "color:default;">Channels in this category boast the highest likes and visits, yet maintain a humble subscriber count. They epitomize high engagement but wrestle with retention rates, creating a vibrant but fleeting viewership.</p>
        </div>

        <p></p>

        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size: 15px;"><p>Eager to see your recommendations? get them <a href = "http://localhost:8502/#enter-your-channel-metrics" target = "_self"><b>Here</b></a>!</p>
        <p>But if you're loving the narrative, click the button below to see some other really cool insights we uncovered 👇🏾</p></div>

        <p></p><p></p>
        """,
        unsafe_allow_html=True
    )


    with st.expander('Click to see more Continent Info!🌍'):
        st.write(
            """
            #### How do content preferences vary across continents?:
            """
        )

        preferences_across_continents(data)

    st.markdown(
        """
        <p></p><p></p><p></p>
        """,
        unsafe_allow_html=True
    )


    st.write(
        """
        ### **Results from Model Training & Validation**
        ---

        """
    )

    st.write(
        """
        In case you wondered, the model is a **`Random Forest Regressor`** from Scikit-Learn's **`sklearn.ensemble`** class, and thanks to some advanced hyperparameter-tuning with **`GridSearchCV`**, we were able to produce some pretty robust results! 💯
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

        <p>To view the data, as well as other assets for the project, check them out on Zahemen's GitHub <a href = "https://github.com/zahemen9900/YouTube-Analytics-App">Here</a></p>
        <p>Thanks for your time!</p></div>

        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
