# # import pandas as pd
# # from scipy.stats import spearmanr
# #
# # # Load the Excel file containing the data
# # df = pd.read_excel('D:/pythonProject2/twitter/excel/新建 XLSX 工作表.xlsx')
# #
# # # Define a function to assign sentiment scores of -1 to emotions less than zero and 1 to emotions greater than zero
# # def assign_sentiment_score(x):
# #     if x < -0.1:
# #         return -1
# #     if x > 0.33334:
# #         return 1
# #     else:
# #         return 0
# #
# # # Define a function to assign comment sentiment score based on the comment count
# # def assign_comment_sentiment_score(x):
# #     if x > 9:
# #         return 3
# #     elif x > 3:
# #         return 2
# #     elif x > 0:
# #         return 1
# #     else:
# #         return 0
# #
# # # Define a function to assign like sentiment score based on the like count
# # def assign_like_sentiment_score(x):
# #     if x > 9:
# #         return 3
# #     elif x > 3:
# #         return 2
# #     elif x > 0:
# #         return 1
# #     else:
# #         return 0
# #
# # # Filter out rows with 0 Likes and apply the function to the Like column in df
# # df_like = df[df['Likes'] > 0]
# # df_like['Like Sentiment Score'] = df_like['Likes'].apply(assign_like_sentiment_score)
# # df_like['Sentiment Score'] = df_like['Score'].apply(assign_sentiment_score)
# #
# # # Filter out rows with 0 Comments and apply the function to the Comment column in df
# # df_comment = df[df['Comments'] > 0]
# # df_comment['Comment Sentiment Score'] = df_comment['Comments'].apply(assign_comment_sentiment_score)
# # df_comment['Sentiment Score'] = df_comment['Score'].apply(assign_sentiment_score)
# #
# # # Calculate the Spearman correlation between comment and sentiment score
# # corr_comment_score, _ = spearmanr(df_comment['Comment Sentiment Score'], df_comment['Sentiment Score'])
# # print("Spearman correlation between comment and score: ", corr_comment_score)
# #
# # # Calculate the Spearman correlation between like and sentiment score
# # corr_like_score, _ = spearmanr(df_like['Like Sentiment Score'], df_like['Sentiment Score'])
# # print("Spearman correlation between like and score: ", corr_like_score)
#

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the Excel file containing the data
df = pd.read_excel('D:/pythonProject2/twitter/excel/新建 XLSX 工作表.xlsx')

# Define a function to assign sentiment scores of -1 to emotions less than zero and 1 to emotions greater than zero
def assign_sentiment_score(x):
    if x < -0.1:
        return -1
    if x > 0.1:
        return 1
    else:
        return 0

# Define a function to assign comment sentiment score based on the comment count
def assign_comment_sentiment_score(x):
    if x > 9:
        return 3
    elif x > 3:
        return 2
    elif x > 0:
        return 1
    else:
        return 0

# Define a function to assign like sentiment score based on the like count
def assign_like_sentiment_score(x):
    if x > 9:
        return 3
    elif x > 3:
        return 2
    elif x > 0:
        return 1
    else:
        return 0

# Apply the function to the Score column in df_comment and df_like
df['Sentiment Score'] = df['Score'].apply(assign_sentiment_score)
df['Comment Sentiment Score'] = df['Comments'].apply(assign_comment_sentiment_score)
df['Like Sentiment Score'] = df['Likes'].apply(assign_like_sentiment_score)

# Create a contingency table for Comments and Score
comment_score_table = pd.crosstab(df['Comment Sentiment Score'], df['Sentiment Score'])

# Calculate the chi-square test for independence between Comments and Score
chi2_comment_score, p_comment_score, dof_comment_score, expected_comment_score = chi2_contingency(comment_score_table)
print("Chi-square test for independence between Comment Sentiment Score and Sentiment Score:")
print("chi2 =", chi2_comment_score)
print("p-value =", p_comment_score)

# Create a contingency table for Likes and Score
like_score_table = pd.crosstab(df['Like Sentiment Score'], df['Sentiment Score'])

# Calculate the chi-square test for independence between Likes and Score
chi2_like_score, p_like_score, dof_like_score, expected_like_score = chi2_contingency(like_score_table)
print("Chi-square test for independence between Like Sentiment Score and Sentiment Score:")
print("chi2 =", chi2_like_score)
print("p-value =", p_like_score)




#多项式回归
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the Excel file containing the data
df = pd.read_excel('D:/pythonProject2/twitter/excel/新建 XLSX 工作表.xlsx')


#Define the independent variable and dependent variables
X = df['Score'].values.reshape(-1, 1)
y_likes = df['Likes'].values.reshape(-1, 1)
y_comments = df['Comments'].values.reshape(-1, 1)


# Fit a polynomial regression model for Likes
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg_likes = LinearRegression()
poly_reg_likes.fit(X_poly, y_likes)

# Fit a polynomial regression model for Comments
poly_reg_comments = LinearRegression()
poly_reg_comments.fit(X_poly, y_comments)

# Print the coefficients and intercepts for Likes and Comments
print("Likes regression model coefficients: ", poly_reg_likes.coef_)
print("Likes regression model intercept: ", poly_reg_likes.intercept_)
print("Comments regression model coefficients: ", poly_reg_comments.coef_)
print("Comments regression model intercept: ", poly_reg_comments.intercept_)

import matplotlib.pyplot as plt

# Plot the scatter plot and the fitted curve for Likes
plt.scatter(X, y_likes, color='blue')
plt.plot(X, poly_reg_likes.predict(X_poly), color='red')
plt.title('Polynomial Regression for Likes')
plt.xlabel('Score')
plt.ylabel('Likes')
plt.show()

# Plot the scatter plot and the fitted curve for Comments
plt.scatter(X, y_comments, color='blue')
plt.plot(X, poly_reg_comments.predict(X_poly), color='red')
plt.title('Polynomial Regression for Comments')
plt.xlabel('Score')
plt.ylabel('Comments')
plt.show()

#
#
#
# import numpy as np
# import pandas as pd
#
# # Load the Excel file containing the data
# df = pd.read_excel('D:/pythonProject2/twitter/excel/新建 XLSX 工作表.xlsx')
#
# # Create numpy arrays for Likes, Comments and Score
# likes = np.array(df['Likes'])
# comments = np.array(df['Comments'])
# score = np.array(df['Score'])
#
# # Calculate the correlation matrix
# corr_matrix = np.corrcoef([likes, comments, score])
#
# # Print the correlation matrix
# print("Correlation Matrix:")
# print(corr_matrix)
#
# # Print the correlation coefficients
# print("\nCorrelation Coefficients:")
# print("Likes and Score:", corr_matrix[0,2])
# print("Comments and Score:", corr_matrix[1,2])
# print("Likes and Comments:", corr_matrix[0,1])
