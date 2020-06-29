# Streamlit-based Web Application
#### SolutionMiners

## 1) Overview

![Streamlit](resources/imgs/streamlit.png)

This repository hosts code to deploy [Streamlit](https://www.streamlit.io/) web application that classifies user input text related to global warming as either as either Pro ,Nuetral ,Anti or news. It also has a natuaral language processing feature.

  
#### **The structure of the app is as follows**
#### 1. Predict Tab
In the side bar drop down ,the user can choose the prediction tab .On this page ,the user is prompted to enter the text they wish to classify , order to determind if the it is either Pro ,Nuetral ,Anti or news.To classify this text ,there are 3 models to choose from which have been trained on the same data set and then prickled to deploy on the app.
The user has the option of Logistic regression, Random forest, or a Support vector machine learning model.After selecting a model,and upon  pressing the "classify" button ,the text is classified by the model of choice ,and the result is returned on screen with class and explaination.
https://github.com/Classification-Team-JM1/Classification_Predict_Package/blob/streamlit_resources/classification-predict-streamlit-jm/resources/imgs/Classes_discription.png

#### 2 .Purpose of the App 
On this tab ,the user can read about why this app was develpoed and how it can be benefitial to them.There is also some brief information as to
why knowing about how people general feel about man-man climate change can be beneficial to them.

#### 3 .Exploratory data Analyses 
Here the user gets to see the data used to develop the machine learning models in the app in the form of cool visual .
The user gets to see the raw data set as it was collected and the final dataset .


##### 4.Machine Learning Tab 
On this tap ,the user can read about the tree machine learning models in the app ,how they work ,how they were trained and how well they performed during training .The user is offered a brief introduction to Logistic Regression ,Random forest and Support vector machine learning algorithm and how they make a prediction .There are also links to sklearn documentation ,which are more technical.


##### 5. Global Warming in 5 minutes

On this tap ,the user can read about Global warming ,what it is and what scientific evidence support that it exist and how it has changed over the years

#### 6.Natural Language Processing 
Here the user gets a glimse of how a computer is able to process natual language .

For this repository, the most crucial file:

| File Name                |   Description                       |
| :---------------------   | :--------------------             |
| `Solution_miners_app.py`          | Streamlit application definition. |

## 2) Usage Instructions
The uses an API to run on aws


#### 2.2) Running the Streamlit web app on your local machine
With the Solution_Minors python script in this repo,the app can be ran on a local machine 

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

 1. Ensure prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 ```

 ```bash
 git clone https://github.com/{your-account-name}/classification-predict-streamlit-template.git
 ```  

 3. Navigate to the base of the cloned repo, and start the Streamlit app.

 ```bash
 change directory to where the repo was cloned 
 streamlit run base_app.py
 ```

 If the web server was able to initialise successfully, the following message is displayed within your bash/terminal session:

```
  The Streamlit app can now be viewed in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```

You should also be automatically directed to the base page of your web app:

![Streamlit base page](resources/imgs/streamlit-base-splash-screen.png)

Congratulations! You've now officially deployed the application!

While we leave the modification of your web app up to you, the latter process of cloud deployment is outlined within the next section.  

#### 2.4) Accessing App via any device with internet access
The web app will be running on a remote EC2 instance 
Follow URL : http://3.250.50.104:500

The user will,in thier web browser of choice , navigate to the URL link above and  should get the corresponse to the following form:

    `http://{public-ip-address-of-remote-machine}:5000`   

    Where the above public IP address corresponds to the one given to your AWS EC2 instance.

    If successful, you should see the landing page of your streamlit web app:

![Streamlit base page](resources/imgs/streamlit-base-splash-screen.png)





