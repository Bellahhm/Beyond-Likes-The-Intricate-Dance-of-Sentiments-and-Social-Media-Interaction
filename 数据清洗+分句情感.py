import xlwt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm # Import tqdm for progress bar

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define regular expression patterns to remove unwanted characters from text
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

# Load the Excel file containing the data
df = pd.read_excel('D:/pythonProject2/twitter/excel/新建 XLSX 工作表.xlsx')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Remove URLs, mentions, and unwanted characters from text
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'http\S+', '', x))
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'@[\w_]+', '', x))
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'\b(rt|im|cz|na)\b', '', x))
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'[^\w\s]|_', '', x))

# Convert text to lowercase and remove stop words
stop_words = set(stopwords.words('english'))
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in stop_words]))
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: x.lower())
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.lower() not in stop_words]))

# Create a list to store the compound scores for each sentence
compounds = []
num_po = 0
num_neg = 0

# Loop through each row in the data, calculate the sentiment score for the row's Embedded_text, and add it to the compounds list
for i, row in tqdm(df.iterrows(), total=len(df)):
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', row['Embedded_text'])  # Remove HTML tags
    sentence = re.sub(r4, '', sentence)  # Remove unwanted characters
    vs = analyzer.polarity_scores(sentence)
    compound = vs["compound"]
    if compound > 0.33333:
        num_po = num_po + 1
    if compound < -0.33333:
        num_neg = num_neg + 1
    compounds.append(compound)

# Calculate the overall sentiment score by taking the average of the compounds list
compounds_all = sum(compounds)
compounds_average = compounds_all / len(compounds)
rate_neg = num_neg / (num_po + num_neg)

# Add a new column to the DataFrame to store the sentiment scores for each row
df['Score'] = compounds

# Save the modified DataFrame to a new Excel file
df.to_excel('D:/pythonProject2/twitter/excel/新建 XLSX 工作表 - 副本.xlsx', index=False)