#Pakages for Visuals
pip install spacy
python3 -m spacy download en_core_web_sm 
import altair as alt
import plotly.figure_factory as ff
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
#Images Dependences

from PIL import Image

# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# NLP Packages
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
#nlp = spacy.load('en')

raw = pd.read_csv("Images/train.csv.csv")
clean = pd.read_csv("Images/clean")
img3 = Image.open('Images/banksy.jpg')

def get_keys(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	
	# Creating sidebar with selection box 
	options = ["Prediction","Purpose of the App", "Exploratory Data Analysis", "About Global Warming", "Machine Learning Models" ,"Natural Language Processing"]
	selection = st.sidebar.selectbox("Choose Option", options)

	if selection == "Exploratory Data Analysis" :
			df_senti1 = raw[raw['sentiment']==1]
			tweet_senti1 = " ".join(review for review in df_senti1.message)

			#create word cloud in eda
			
			st.image(img3 ,width = 600, caption = "Visualising the climate change threat")

			st.title("Insight From The Data")
			st.subheader("A Representation Of The Most Common Words In Each Sentiment Class")
			sent_groups = st.radio('Sentiment Views:',('Positive, those who believe climate change is a threat',
			'Negative sentiment, opposing the belief that climate change is a threat', 'Neutral, an impartial stance on climate change', 
			'News Report, topical news reported on climate change'))
			if sent_groups == ('Positive, those who believe climate change is a threat'):
				df_senti1 = clean[clean['sentiment']==1]
				tweet_senti1 = " ".join(review for review in df_senti1.clean_stp_words)
				# Create and generate a word cloud image:
				wordcloud_1 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_senti1)
				plt.imshow(wordcloud_1, interpolation='bilinear')
				#plt.set_title('Tweets under Pro Class 1',fontsize=50)
				plt.axis('off')
				plt.show()
				st.pyplot()
				if st.checkbox('Interpretation of Diagram, Sentiment 1'):
					"""Common words of interest in pro-sentiment include `To fight`,`to tackle`, `belive in` and `fight climate`. It appears that 
					tweets in this category are providing solutions to fight climate change. Many of the sentiments reflected are related to 
					on Trumps commentary. In the pro sentiment class we find that people do not agree with Trump.')"""
			if sent_groups == 'News Report, topical news reported on climate change':
				df_senti_2 = clean[clean['sentiment']==2]
				tweet_senti_2 = " ".join(review for review in df_senti_2.clean_stp_words)
				# Create and generate a word cloud image:
				wordcloud_2 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_senti_2)
				plt.imshow(wordcloud_2, interpolation='bilinear')
				#plt.set_title('Tweets under Pro Class 1',fontsize=50)
				plt.axis('off')
				plt.show()
				st.pyplot()
				if st.checkbox('Interpretation of Diagram, Sentiment 2') :
					"""Common words news tweets are `Trump, global warming, via`,`Scientists`,`researchers`,`ÈPA` and `report`.
				 	This could reveal the sentiment that humans are the cause of climate change because they burn fossil fuels. 
				 	News reports can be highly influential on overall sentiment as many rely of the media to validate their beliefs. 
				 	It is evident that the word Trump is  most common. According to research in the news, the momentum for these 
				 	sentiments comes from the commentary that president Trump has made about climate change."""

			if sent_groups == "Neutral, an impartial stance on climate change":
				df_senti_0 = clean[clean['sentiment']==0]
				tweet_senti_0 = " ".join(review for review in df_senti_0.clean_stp_words)
				#Create and generate a word cloud image:
				wordcloud_0 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_senti_0)
				plt.imshow(wordcloud_0, interpolation='bilinear')
				#plt.set_title('Tweets under Pro Class 1',fontsize=50)
				plt.axis('off')
				plt.axis("off")
				plt.show()
				st.pyplot()
				if st.checkbox('Interpretation of Diagram, Sentiment 0'):
					"""The sentiments in class 0 represents people that are neutral towards climate change. The reason could be that they are
					not aware of climate change, or do not have enough information, this can be seen by words such as `interviewer`,
					`Trump`, `think`. Common words in neutral tweets include `care about`,`think`,`maybe`. This could indicate
					uncerainty toward climate change validity or an apathetic inclination.Interestingly, the appearance of the word
				`	ignore` tells us that these tweeters find the matter confusing.')"""
			
			st.subheader("**Observe the frequency of the 20 most common words in each class**")
			Pro = clean[clean['sentiment']==1]
			Anti = clean[clean['sentiment']== -1]
			Neutral = clean[clean['sentiment']==0]
			News = clean[clean['sentiment']== 2]

			common = st.selectbox('Select Sentiment Type',('Positive','Negative', 'Neutral','News'))
			if common == 'Positive':
				Pro['temp_list'] = Pro['clean_stp_words'].apply(lambda x:str(x).split())
				top = Counter([item for sublist in Pro['temp_list'] for item in sublist])
				temp_positive = pd.DataFrame(top.most_common(20))
				temp_positive.columns = ['Common_words','count']
				temp_positive = temp_positive.style.background_gradient(cmap='Greens_r')
				st.write(temp_positive, width=200)
			if common == 'Negative':
				Anti['temp_list'] = Anti['clean_stp_words'].apply(lambda x:str(x).split())
				top = Counter([item for sublist in Anti['temp_list'] for item in sublist])
				temp_neg = pd.DataFrame(top.most_common(20))
				temp_neg.columns = ['Common_words','count']
				temp_neg = temp_neg.style.background_gradient(cmap='Greens_r')
				st.write(temp_neg, width=200)
			
			if common == 'News':
				News['temp_list'] = News['clean_stp_words'].apply(lambda x:str(x).split())
				top = Counter([item for sublist in News['temp_list'] for item in sublist])
				temp_news = pd.DataFrame(top.most_common(20))
				temp_news.columns = ['Common_words','count']
				temp_news = temp_news.style.background_gradient(cmap='Greens_r')
				st.write(temp_news, width=200)

			if common == 'Neutral':
				Neutral['temp_list'] = Neutral['clean_stp_words'].apply(lambda x:str(x).split())
				top = Counter([item for sublist in Neutral['temp_list'] for item in sublist])
				temp_net = pd.DataFrame(top.most_common(20))
				temp_net.columns = ['Common_words','count']
				temp_net = temp_net.style.background_gradient(cmap='Greens_r')
				st.write(temp_net, width=200)

			st.subheader("**A Closer Look At The Data Distribution**")
			temp = raw.groupby('sentiment').count()['message'].reset_index().sort_values(by='message',ascending=False)
			temp['percentage'] = round((temp['message']/temp['message'].sum())*100,0)
			labels1 = temp['sentiment']
			labels = ["Sentiment  %s" % i for i in temp['sentiment']]
			sizes = temp['percentage']
			fig1, ax1 = plt.subplots(figsize=(6, 6))
			fig1.subplots_adjust(0.3, 0, 1, 1)

			theme = plt.get_cmap('Greens_r')
			ax1.set_prop_cycle("color", [theme(1. * i / len(sizes))
			                         for i in range(len(sizes))])
			_, _ = ax1.pie(sizes, startangle=90, labels = labels1,  radius=1800)

			ax1.axis('equal')
			total = sum(sizes)
			plt.legend(
			loc='upper left',
			 labels=['%s, %1.1f%%' % (
				l, (float(s) / total) * 100)
				for l, s in zip(labels, sizes)],
			prop={'size': 7},
			bbox_to_anchor=(0.0, 1),
			bbox_transform=fig1.transFigure)
			
			plt.show()  # Equal aspect ratio ensures that pie is drawn as a circle.
			st.pyplot()   #c, use_container_width=True)

			if st.checkbox('Interpretation of Pie Chart'):
				"""More than half of the tweets analysed reflect a belief in climate change. 
					Although it is not an overwhelming majority figure, believers are in the majority.
					As science begins to offer clearer evidence it is likely that many neutral tweeters 
					could sway their beliefs. Less than ten percent of the sample population do not believe 
					in climate change. If the sample is a good representation of the population than the
					market for evironmentally friendly or environmentally conscious goods and services could
					be a desireable product to fairly large sector of the population')"""
				

				
            	
	if selection == "Purpose of the App" :

			st.header("**The Impact Of Climate Change Sentiment And Maximising Profit**")
			img2 = Image.open("Images/gw.jpeg.jpg")
			st.image(img2 ,width = 400, caption = "Visualising the climate change threat")
			"""This app will reveal the overall sentiment toward climate change by analysing recent
			tweets (post made on the social media application Twitter).By understanding how potential consumers 
			view climate change, companies can make informed decisions on product development and marketing. This app
			 will answer the question: Do people see climate change as a real threat?"""
			
			st.subheader("A brief Look At The Raw Data (Database of tweets analysed)")
			
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[["sentiment","message"]]) # will write the df to the page
				data = pd.DataFrame(raw, columns = ['sentiment', 'message'] )
				st.write(data.plot(kind = 'hist', color = 'green'))
				st.pyplot()
				data = {'Sentiment Type': ['-1', '0', '1', '2'], 'Sentiment Meaning':
				 ['Negative sentiment, opposing the belief that climate change is a threat', 'Neutral, an impartial stance on climate change',
				 'Positive, supporting the belief that climate change poses a threat', 'News Report, topical news reported on climate change'] }
				sentiment = pd.DataFrame(data, columns = ['Sentiment Type', 'Sentiment Meaning'])
				sentiment = sentiment.set_index('Sentiment Type')
				st.write(sentiment, width = 800)
				
				st.subheader("**Interpretation Of Sentiment Distribution**")
				"""In the database ,most of the tweets indicate that alot of people believe climate change is a real threat and is man-man."""
				"""Media coverage on climate change concerns substantiates the belief that climate change is a real threat.There are tweets 
				in the database that indicate that there are people who are nuetral on the subject of the subject
			    of Global warming ,however ,they are vastly outnumbered"""
			

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

			logistic_regression = Image.open("Images/logistic_regression.jpg")
			st.image(logistic_regression, caption = "sigmoid function for logistic regression ",use_column_width=True)
			"""For multiclass classification problems ,
			logistic regression models are combined into what is known as the one-vs-rest approach (or OvR).
			In the OvR case, a separate logistic regression model is trained for each label that the response
			variable takes on."""
			st.subheader("Pros and cons of Logistic Regression")
			""" - easy to implement and very efficient to train"""
			""" - Can overfit when data is unbalanced and Doesn't handle large number of categorical variables well."""
			logistic_reg_perf = Image.open('Images/logistic_reg_perfomance.jpg')
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
			decisiontree = Image.open("Images/random_forest.png")
			st.image(decisiontree, caption = "Random Forest tree process to predict a label ",width=None)
			 
			st.subheader("Pros and cons of the random forrest")
			""" - Can handle missing values well. Missing values are substituted by the variable appearing the most in a particular node."""
			""" - Provides the some of the highest accuracy of available classification methods"""
			""" - Some drawbacks is that the random forst classifyer method is that it requires a lot of computational reasources
			 time consuming ,and less intuitive compared to other algorithms"""
			random_for_perf =Image.open("Images/random_forest_perf.jpg")
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
			svm = Image.open("Images/support_vector1.jpg")
			st.image(svm,caption = "Hyperplane deviding data points" ,use_column_width=True)
			st.subheader("Pros and Cons of Support Vector Machines")
			"""- it is very accurate and works well on smaller cleaner datasets"""
				
			""" - It can be more efficient because it uses a subset of training points"""
			""" - Less effective on noisier datasets with overlapping classes , 
			training time with SVMs can be high ,thus not suitable for larger datasets"""
			svm_perf = Image.open("Images/support_vector_perfomance.jpg")
			st.image(svm_perf,use_column_width=True)
			st.header("For more information on algorithm implimentation")
			"**Logistic regression**"
			" https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
			"**Random Forest **"
			
			" https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
			"**Support Vector Machines** "
			" https://scikit-learn.org/stable/modules/svm.html"



    
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

	if selection == "About Global Warming":
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

		image = Image.open("Images/global temperature.jpg")
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
				predictor = joblib.load(open(os.path.join("resources/saved_model_for_App.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
			elif model_choice == "Support Vector Machine":
				predictor = joblib.load(open(os.path.join("resources/saved_model_for_App.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
				# st.write(prediction)
			elif model_choice == "Random Forest Tree":
				predictor = joblib.load(open(os.path.join("resources/saved_model_for_App.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])

			#Results displayed on screen after User has clicked the classify button
			final_result = get_keys(prediction,prediction_labels)
			st.success("{}".format(final_result))

#Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
