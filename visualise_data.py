import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv("banking.csv")

data['education'] = np.where(data['education']=='basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.4y', 'Basic', data['education'])

sns.countplot(x='y',data=data,palette='hls')
plt.show()
plt.savefig('count_plot')

count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])
pct_of_no_sub = count_no_sub/(count_sub+count_no_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_sub+count_no_sub)
print("percentage of subscription is", pct_of_sub*100)
print("ratio of subscriptions to no_subscriptions", int(pct_of_no_sub*100), ":", int(pct_of_sub*100))

pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.show()
plt.savefig('purchase_frequency_job')

table = pd.crosstab(data.education,data.y).plot(kind='bar')
plt.title('Bar chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('No of subscriptions')
plt.savefig('edu_vs_pur_stack')
plt.show()

table = pd.crosstab(data.marital,data.y).plot(kind='bar')
plt.title('Bar chart of Marital Status vs No of Customers')
plt.xlabel('marital status')
plt.ylabel('No of customers')
plt.savefig('mar_vs_customers')
plt.show()

table = pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title("Purchase frequency for day of Week")
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('per_day_week_purchase')
plt.show()

table=pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase frequency for month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.show()
plt.savefig('monthly_customers')

data.age.hist()
plt.table('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
plt.savefig('hist_age')


