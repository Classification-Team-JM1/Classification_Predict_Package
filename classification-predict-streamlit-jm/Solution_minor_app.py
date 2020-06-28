"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# NLP Packages
import spacy
nlp = spacy.load('en')

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

#Word cloud and Images
from wordcloud import WordCloud
from PIL import Image
raw = pd.read_csv(r"C:\Users\chari\Documents\Explore\classification-predict-streamlit-template\resources\train.csv")

def get_keys(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")
	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction","Exploratory Data Analysis", "Information","Machine Learning Models" ,"Natural Language Processing"]
	selection = st.sidebar.selectbox("Choose Option", options)


	if selection == "Machine Learning Models" :
	

			st.header("**Logistic Regression**")
			"""The Logistic regression algorithm builds a regression model to predict
			the probability that a given data entry belongs to the category numbered as “1”.
			Logistic regression becomes a classification technique only when a decision
			threshold is brought into the picture. The setting of the threshold value is a very 
			is dependent on the classification problem itself.
			Logistic regression models the data using the sigmoid function.
			It squeezes the range of output values to exist only between 0 and 1.
			For binary classification ,the output value of a logistic regre.The threshold 
			value is usually set to 0.5 and determine if an observation will belong to class 0 or 1."""
			logistic_regression = Image.open(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\Images\logistic_regression.jpg")
			st.image(logistic_regression, caption = "sigmoid function for logistic regression ",use_column_width=True)
			"""For multiclass classification problems ,
			logistic regression models are combined into what is known as the one-vs-rest approach (or OvR).
			In the OvR case, a separate logistic regression model is trained for each label that the response
			variable takes on."""
			st.subheader("Pros and cons of Logistic Regression")
			""" - easy to implement and very efficient to train"""
			""" - Can overfit when data is unbalanced and Doesn't handle large number of categorical variables well."""
			logistic_reg_perf = Image.open(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\Images\logistic_reg_perfomance.jpg")
			st.image(logistic_reg_perf,use_column_width=True)

			st.header("**Random Forest tree**")
			"""The building blocks of the random first model are Decision trees.Simple put ,the decision tree is a flowchart
			 of questions leading to a prediction.
			Random forest is a technique used in modeling predictions and behavior analysis and is built on decision trees.
			It contains many decision trees that represent a distinct instance of the classification of data input into the random forest. 
			The random forest technique takes consideration of the instances individually, taking the one with the majority of votes as 
			the selected prediction."""

			"""Each decision tree in the forest considers a random subset of features when forming questions and only has access
				to a random set of the training data points.This increases diversity in the forest leading to more robust overall predictions and the name
				 ‘random forest.’ When it comes time to make a prediction, the random forest takes an average of all the individual decision tree estimates
				"""

			"""Each tree in the classifications takes input from samples in the initial dataset.This is followed by a random selection of Features 
			(or indipendent variables) , which are used in growing the tree at each node. Every tree in the forest is pruned until 
			 the end of the exercise when the prediction is reached decisively. 
			Thus ,the random forest enables any classifiers with weak correlations to create a strong classifier"""
			decisiontree = Image.open(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\Images\random_forest.png")
			st.image(decisiontree, caption = "Random Forest tree process to predict a label ",width=None)
			 
			st.subheader("Pros and cons of the random forrest")
			""" - Can handle missing values well. Missing values are substituted by the variable appearing the most in a particular node."""
			""" - Provides the some of the highest accuracy of available classification methods"""
			""" - Some drawbacks is that the random forst classifyer method is that it requires a lot of computational reasources
			 time consuming ,and less intuitive compared to other algorithms"""
			random_for_perf =Image.open(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\Images\random_forest_perf.jpg")
			st.image(random_for_perf ,use_column_width=True)
			
			st.header("Support Vector Machine")

			"""A Support Vector Machine (SVM) is a supervised machine learning algorithm that can be employed for both 
			classification and regression purposes.SVMs are based on the idea of finding a hyperplane that best divides 
			a dataset into two classes"""
			"""Support vectors are the data points nearest to the hyperplane, the points of a data set that, if removed, would alter 
			the position of the dividing hyperplane. Because of this, they can be considered the critical elements of a data set."""
			"""Simply put ,a hyperplane is a line that linearly separates and classifies a set of data."""
			"""The further from the hyperplane a data point lies, the higher the probability that it has been 
			classified correctly. Ideally ,we require a data point to be as far away as possible , while still being on 
			the correct side of the hyperplane .Whenever new testing data is added ,the side of the hyperplane is 
			lands on decides the class it is assigned to.
			"""
			svm = Image.open(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\Images\support_vector1.jpg")
			st.image(svm,caption = "Hyperplane deviding data points" ,use_column_width=True)
			st.subheader("Pros and Cons of Support Vector Machines")
			"""- it is very accurate and works well on smaller cleaner datasets"""
				
			""" - It can be more efficient because it uses a subset of training points"""
			""" - Less effective on noisier datasets with overlapping classes , 
			training time with SVMs can be high ,thus not suitable for larger datasets"""
			svm_perf = Image.open(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\Images\support_vector_perfomance.jpg")
			st.image(svm_perf,use_column_width=True)
			st.header("For more information on algorithm implimentation")
			"**Logistic regression**"
			" https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
			"**Random Forest **"
			
			" https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
			"**Support Vector Machines** "
			" https://scikit-learn.org/stable/modules/svm.html"



    #Natural language Processing page ,it slowed down my computer because of the english library
	if selection == 'Natural Language Processing':
		st.info("Natural Language Processing")
		tweet_text = st.text_area("Enter Text","Type Here")
		nlp_task = ["Tokenization","NER","Lemmatization","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text {}".format(tweet_text))

			docx = nlp(tweet_text)
			if task_choice == 'Tokenization':
				result = [ token.text for token in docx ]
				
			elif task_choice == 'Lemmatization':
				result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
			elif task_choice == 'NER':
				result = [(entity.text,entity.label_)for entity in docx.ents]
			elif task_choice == 'POS Tags':
				st.json(result)

		if st.button("Tabulize"):
			docx = nlp(tweet_text)
			c_tokens = [ token.text for token in docx ]
			c_lemma = [token.lemma_ for token in docx]
			c_pos = [word.tag_ for word in docx]

			new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(new_df)

		if st.checkbox("Wordcloud"):
            
			wordcloud =  WordCloud(max_font_size=30, max_words=100, background_color="orange").generate(tweet_text)
			plt.imshow(wordcloud,interpolation='bilinear')
			plt.axis("off")
			st.pyplot()
		
	
	# Building out the "Information" page
	

	if selection == "Information":
		st.info("General Information")
		
		""" # Global Warming in 5 minutes """
		st.header("Natural Climate change")
		""" - Throughout its long history, Earth has warmed and cooled time and again.
		Climate has changed when the planet received more or less sunlight due to subtle shifts
		 in its orbit, as the atmosphere or surface changed, or when the Sun’s energy varied .This was
		 all without any help from humanity"""
		""" - Earth’s temperature begins with the Sun. Roughly 30% of incoming sunlight is 
		reflected back into space by bright surfaces like clouds and ice. The rest is absorbed by
		 the land and ocean, and the atmosphere. 
		 The absorbed solar energy heats our planet and makes it habitable."""
		""" - As the rocks, the air, and the seas get warmer, they radiate “heat” energy which 
		 travels into the atmosphere ,where it is absorbed by water vapor and long-lived greenhouse 
		 gases """

		""" - Greenhouse gases are those gases in the atmosphere that have an influence on the earth's energy balance. 
		The best known greenhouse gases, carbon dioxide (CO₂), methane and
		 nitrous oxide, can be found naturally in low concentrations in the atmosphere.
		"""	 
		""" - After absorbing the heat energy ,these greenhouse gases will radiate energy in all directions. Some of this energy is 
		 radiated back towards the Earth ,further warming atmosphere and surfaces - This is the natural greenhouse"""

		""" - Some natural forces that contribute to climate change include volcanic eruptions, which
		pump out clouds of dust and ash, which block out some sunlight. Volcanic debris also includes sulfur dioxide,
		 combines with water vapor and dust in the atmosphere to form sulfate aerosols, which reflect sunlight away
		  from the Earth’s leading to a cooling effect."""
		""" - Earth orbital changes - Shifts and wobbles in the Earth’s orbit can trigger changes in climate such as 
		the beginning and end of ice ages"""

		""" - Also natural is Solar variations. Although the Sun’s energy output appears constant
		from an everyday point of view, small changes over an extended period of time can lead to climate changes.
		 Some scientists suspect that a portion of the warming in the first half of the 20th century was due to an
		  increase in the output of solar energy"""
		"""- Scientists constantly measure these natural effects, but none can account for the observed trend since 1970.
		 Scientists can only account for recent global warming by including the effects of human greenhouse gas emissions."""

		image = Image.open(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\Images\global temperature.jpg")
		st.image(image, caption="Global temperature graph(Image: Global Warming Art)", use_column_width=True)
		st.subheader("Some notable events in The Global Temperature timeline")
		""" Between 1850-1890 , the Mean global temperature was roughly 13.7°C.This is the time period of the First Industrial Revolution. 
			Coal, railroads, and land clearing speed up greenhouse gas emission, while 
			better agriculture and sanitation speed up population growth."""
		"""Between 1870-1910 was the Second Industrial Revolution. Fertilizers and other chemicals, 
		electricity, and public health further accelerate population growth."""
		""" Around 1940 ,massive output of aerosols from industries and power plants 
		contributed to the global cooling trend from 1940-1970."""
		""" two major volcanic eruptions, El Chichon in 1982 and Pinatubo in 1991, pumped sulfur dioxide gas high into the atmosphere.
		 The gas was converted into tiny particles that lingered for more than a year, reflecting sunlight and shading Earth’s surface
		 causing cooling for two to three years."""
		"""The 10 warmest years on record have all occurred since 1998, and 9 of the 10 have occurred since 2005."""

		"""Models predict that Earth will warm between 2 and 6 degrees Celsius in the next century. When global warming has
		 happened at various times in the past two million years, it has taken the planet about 5,000 years to warm 5 degrees.
		 e predicted rate of warming for the next century is at least 20 times faster"""




		"""- Factuations climate is natural but scientists say temperatures are now rising faster 
		than at many other times."""
		""" - Humans have been artificially raising the concentration of greenhouse gases in the atmosphere ,causing the enhanced
		Greenhouse effect """
		""" - Global warming is the unusually rapid increase in Earth’s average surface temperature over the past 
		century primarily due to the greenhouse gases released as people burn fossil fuels. """
		
		""" - According to IPCC in its 5th 2013 fifth assessment report ,there is 
		between a 95% and 100% probability that more than half of modern day warming was due to humans."""
		""" - Recent US fourth national climate assessment found that between 93% to 123% of observed 
			1951-2010 warming was due to human activities"""
		""" - Human activities like burning fossil fuels leading to higher carbon dioxide concentrations,
			farming and forestry — including land use change via agriculture and livestock
			cement manufacture
			aerosols — chlorofluorocarbons (CFCs) have been linked to Global warming"""

		""" - Greenhouse gases from these activities collect in the atmosphere and absorb sunlight and 
		solar radiation that have bounced off the earth’s surface. Normally, this radiation would escape 
		 into space—but these pollutants, which can last for
		 years to centuries in the atmosphere, trap the heat and cause the planet
		  to get hotter. That's what's known as the greenhouse effect """
		"""- There are Natural external causes such as increases or decreases in volcanic activity or solar radiation.
		 For example, every 11 years or so, the Sun’s magnetic field flips ,this can cause small 
		 fluctuations in global temperature, up to about 0.2 degrees. On longer time scales – tens to hundreds
		  of millions of years – geological processes can drive changes in the climate, due to shifting
		   continents and mountain building"""


		""" # Evidence of Global Warming 📈 """
		""" - Across the globe, average sea level increased by 3.6mm per year between 2005 and 2015 """ 
		""" - According to the World Meteorological Organization (WMO),The world is about one degree Celsius warmer 
		than before widespread industrialisation"""
		""" - Data from NASA's Gravity Recovery and Climate Experiment show 
		The Greenland and Antarctic ice sheets have decreased in mass"""
		st.subheader("Suggested Readings :earth_africa: ")
		st.markdown("https://www.bbc.com/news/science-environment-24021772")
		st.markdown("https://climate.nasa.gov/evidence/")
		st.markdown("https://earthobservatory.nasa.gov/features/GlobalWarming/page2.php")
		st.markdown("https://www.carbonbrief.org/analysis-why-scientists-think-100-of-global-warming-is-due-to-humans")
		# read in word file

		st.subheader("Refereces")
		"""1.https://www.newscientist.com/article/dn11639-climate-myths-the-cooling-after-1940-shows-co2-does-not-cause-warming/"""


		st.subheader("Climate change tweet classification")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text to Classify ","Type Here")
		#tweet_text=[tweet_text]
		#tweet_text = st.text_area("Enter Text","Type Here")
		all_ml_models = ["Logistic Regression","Support Vector Machine","Random Forest Tree"]
		model_choice = st.selectbox("Choose ML Model",all_ml_models)
		prediction_labels = {"Neutral : This text neither supports nor refutes the belief of man-made Climate change":0,
		"Pro : This text shows belief in man-man climate change":1,
		"news : This text is links to factual news about climate change":2,
		"Anti : This text shows lack of belief in man-made climate change":-1}
		if st.button("Classify"):
			if model_choice == "Logistic Regression":
				predictor = joblib.load(open(os.path.join(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\saved_model_for_App.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
			elif model_choice == "Support Vector Machine":
				predictor = joblib.load(open(os.path.join(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\saved_model_for_App.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
				# st.write(prediction)
			elif model_choice == "Random Forest Tree":
				predictor = joblib.load(open(os.path.join(r"C:\Users\chari\Desktop\mln\classification-predict-streamlit-template\resources\saved_model_for_App.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

			#Results displayed on screen after User has clicked the classify button
			final_result = get_keys(prediction,prediction_labels)
			st.success("{}".format(final_result))

#Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
