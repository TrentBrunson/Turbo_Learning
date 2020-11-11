# University of Texas Defensive Play-Call Predictor


## Team Information 
### Members: 
* Trent Brunson
* Pablo Maldonado 
* Maricela Avila 
* Travis Greenfield 

### Team Communication Plan 
* Branches for the first week will be named using each member's respective first name, the following weeks the format will consist of first name and task description:
  * EX: Maricela_XXInsert Task DescriptionXXX
* Each team member will make commits on their branch
* Each team member is responsible for letting the merging lead (Trent) know when there is a pull request outstanding for their respective section
* Trent will serve as lead to make the final weekly merge

## Project

### Selected Topic 
We will analyze college football (CFB) data from several years to determine what defense a team should call based on past successes and failures. 

### Reasoning for Topic 
In football, good defense keeps the other team from scoring and provides a good field position, which allows the offense to start their drive-to-goal in a better position, thus allowing an easier time scoring. 

### Data Source Description
We obtained reliable CFB data from the ESPN API, allowing us to analyze several years of data, which we have narrowed down to 6 years. Based on the amount of data we have, we believe these years are enough to accurately identify trends. 

### Question The Team Hopes To Answer:
After completing the project, we hope to be able to answer the following question: 
* What is the likelihood that the opposing team will call on a "run" or "pass" based on the play, distance or time?
* Are there any variables that have a greater affect on the decision call? If so, what are they? 

## Segment 2

### Presentation 
Please view our ongoing team presentation draft here: https://docs.google.com/presentation/d/1fYbRv5cNzHDDJVVBG5kk5iRclJ1ldfzgQI6HdRqE9GY/edit?usp=sharing

The presentation outlines the project, including the following: 
Selected topic
* Reason why we selected our topic 
* Description of our source of data
* Questions we hope to answer with the data
* Description of the data exploration phase of the project
* Description of the analysis phase of the project

### Database
For our database, we used **Heroku** to store our data. Our decision to use this cloud platform was due to the dynamic features and its free of cost offerings. 

Detailed below is our ERD:

![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/DB/QuickDBD-CFB_DB.png)

During our data exploration, we identified a primary key, joined tables to more accurately represent the data, dropped duplicates and narrowed the number of years and attributes to test, based of what we found to be of highest importance. 

![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/Database-1.PNG)


### Dashboard 
For this segment, we have completed a storybaord of what will become our dashabord, listed our tools and determined the interactive features. See below for a full description of our storyboard and working plan. 

#### Tools Used:
* HTML
* Bootstrap
* CSS
* Heroku

#### Outline

![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/Dashboard-1.PNG)

1.) Landing Page
  * Clean and interactive 
  * Contains a navigation bar linking to the Methodology and Machine Learning Model Findings
  * Top section explans the purpose of the dashboard and entices users to use our predicting model 
  
2.) User Input Sidebar
  * In a left sidebar, the user input fields allow for data entry. Once data is entered, the user presses the “Call Play” button to activate our ML model, allowing the user to   predict the next defensive play call

![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/Dashboard-2.PNG)

3.) Results displayed 
  * After the user prompts a prediction from the ML model by pressing the “Call Play” button, a probability of plays will be shown via a heatmap chart
  * Color coding will allow users to have a more user-friendly experience and fast response 

![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/Dashboard-3.PNG)

4.) For the data geek, we included a Methodology page for users to learn more about the model used, its outcomes and limitations, as well as data processing and development. 

![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/Dashboard-5.PNG)

5.) Lastly, we’ve included a Findings page dedicated to the evaluation of the model and recommendations for future development.

### Machine Learning & Analysis

* **Preprocessing** - coded in ‘clean.py’
  * Used the ‘glob’ method to combine years of data into a single dataframe.
  * Initial columns kept: 'gameId', 'driveIndex', 'playIndex', 'year', 'week', 'homeAbbr', 'awayAbbr', 'offenseAbbr', 'defenseAbbr', 'homeScore', 'awayScore', 'quarter', 'clock',    'type', 'down', 'distance', 'yardLine', 'yardsGained'.
  * Dropped N/A values along with one duplicate discovered in the ‘2014’ row.
  * Transformed related columns to be specific to their relation to Texas -- aligning with the stated goal of prediction plays of Texas’ opponents.
  * Plays that did not result in a Rush or Pass (i.e. penalties, timeouts, etc.) were removed.

* **Feature Engineering** - coded at the top of ‘playcall_ml.py’
  * Converts ‘clock’ from HH MM SS datetime format to an integer of total seconds.
    * Allows us to bin the results or treat as a continuous value.
    * Converted to ‘seconds_in_quarter_remaining’, used the ‘quarter’ value to convert to ‘seconds_in_half_remaining’, and ultimately binned to 'time_remaining_binned'
      * Chose 2 minute intervals to account for the fact that strategies around the 2 minute remaining mark likely change.
  * Used ‘quarter’ value to create ‘half’ feature.
  * Converts all outcomes of Pass/Rush (i.e. incomplete pass, passing touchdown, rushing touchdown, etc.) to Pass or Rush.
    * This creates a true binary and also accounts for the fact that we’re not interested in the outcome outside of the play chosen.
  * Continuous features (‘yards’, ‘down’) are standardized using the StandardScaler method from sklearn.
  * Categorical and Object datatyped features or encoded using the OneHotEncoder method from sklearn.
      * Subsequent columns are then remerged with the original dataset and the original columns are dropped.
      
* **Feature Analysis and subsequent selection / removal** - *decisions on which columns make it to the feature dataframe and which are ultimately dropped. 
Code in ‘feature_analysis.py’ and ‘featureMap.py’.
  * Feature ‘texscore’ & ‘oppscore’ (used) - in conjunction, these two features help determine if a team needs to score quickly -- leading to more passing -- or whether they         want to ‘run out the clock’ -- leading to more run plays.
  * Feature ‘clock’ transformed into ‘time_remaining_in_half” (used) - This allows us to both bin easily and treat it as a continuous feature should it be more beneficial to the 
    model. It also reflects the fact that strategies change as teams get closer to the end of the half and not so much at the end of the quarter.
  * Feature ‘quarter’ and ‘half’ (used) - As the game gets later, strategies will change, and that seems to be reflected in the feature analysis below.
  * Feature ‘down’ and ‘distance’ (used) - A primary contributor (something that’s highlighted as well by the feature analysis below). It’s also likely the main contributor to a 
    naive, non-ML derived decision of rush/pass.
  * Feature ‘yardline’ (used) - As field position changes, the playbook opens up and alters strategy. 
  * Feature ‘week’ (dropped) - While an individual team’s strategy may change week over week as personal, practice time, and coaching philosophies change, that’s not something
    that will stay consistent across teams or years and the inclusion could result in overfitting.
  * Feature ‘year’ (dropped) - Similar to the reasoning behind ‘week’ being dropped. Additionally, in preliminary feature analysis, year wasn’t a huge driver of change.
  
* **Feature Important Analysis** - These results fit well with the information gathered by the Seaborn heatmap. Code written in ‘featuer_analysis.py’ and ‘featureMap.py’
  * **Linear Regression Feature Importance** - By using the LinearRegression method from sklearn, and fitting the data to the model, we reveal that the most impactful features are       ‘quarter’, ‘down’, and ‘distance’ aligning with what we see with the Seaborn Heatmap.
       
       ![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/LinearRegressionFeatureImportance.png)

  * **CART Classification Feature Importance** - A little bit of a different story here with the ‘quarter’, ‘down’, and ‘distance’ being featured less importantly in the model.     With the growth of the other features, however, it could be an indication that they’re worth keeping in the model. 
       
       ![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/CART%20Classification%20Feature%20Importance.png)

  * **RF Classifier Feature Importance** - This is where we’re starting to see some consistency with feature importance. First, it’s notable that a majority of the features    
    chosen are having an impact. ‘half’ may warrant a second look given it’s low score compared to the others. 
       
       ![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/RFClassifierFeatureImportance.png)
       
  * **RF Regression Feature Importance** - Very consistent with the analysis above. The change of note is the increase in ‘quarter’, ‘down’, and to a smaller extent ‘distance’  
     which aligns with what we’re seeing with earlier analysis.
       
       ![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/RFRegressionFeatureImportance.png)
  
* **Train, Test, Split** - how the dataset was split for training and testing purposes.
  * Using the train_test_split method from sklearn, the data was split to allow for **75%** to be used for training, and **25%** to be used to test accuracy, precision, and    
    recall for the model.
  * No need to stratify as the output variables are near 50/50 splits.

* **ML Choice: Random Forest**

  * **Benefits** - Computationally, Random Forests are able to handle large datasets in addition to a wide set of input features. While we’re currently looking at a relatively  
    smaller dataset, we have the ability to expand it without worrying about the model’s accuracy changing. Additionally, RF models aren’t as prone to overfitting as other   
    choices, meaning this model could likely be extended outside of our limited scope.
    
  * **Limitations** - the “black box” issue. While there are options available for feature analysis when preparing the data, once the model is making its decisions, it’s tough       to tell what actions it’s taking. The best way to counteract that is to have good data in. Additionally, RF models aren’t as prone to overfitting as other choices, meaning       this model could likely be extended outside of our limited scope.
  
* **Iterations**
  * 1st - Random Forest, Binned time, non-binned “distance”, kept “week”
    Accuracy score: 0.6272
  * 2nd - Random Forest, Binned time, non-binned “distance”, dropped “week”
    Accuracy score: 0.6368
  * 3rd - Random Forest, Binned time, non-binned “distance”, dropped “week”, dropped “year”
    Accuracy score: 0.6368
    Reverted back to ‘2nd’ RF model.


## Segment 3

### Presentation 
Please view our ongoing team presentation draft here: https://docs.google.com/presentation/d/1fYbRv5cNzHDDJVVBG5kk5iRclJ1ldfzgQI6HdRqE9GY/edit?usp=sharing

### Database
For our database, we finalized our Heroku connection with the dashboard. While we're still finalizing our connection with the flask app, it's coming along. Our data and dhashboard needed to migrated to Heroku in order to be able to read the data we sorted and organized in the cloud platform. 

### Dashboard

This week, the team updated the esthetics and deployed the flask app to continue to make steps in having a fully functional connection with our machine learning module. 
Once connected, the "Call Play" button reacted by displaying the outcome, which is either "Run" or "Pass". 

See below for an updated image on our interactive feature: 
       ![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/MS3-Dashboard_interactive.png)
       

Additionally, we improved the esthetics to be more on theme with UT, and added a probability chart to the right for added information.

       ![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/MS3-Dashboard_Index.PNG)


### Machine Learning
We continued to work for a better accuracy score and finalize elements for a successful connection to the dashboard.        



