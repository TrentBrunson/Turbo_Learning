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

![alt text](https://github.com/TrentBrunson/turbo-learning/blob/main/Presentation%20Images/Dashboard-5.PNG)






