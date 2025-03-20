import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
   
# Read the data from the uploaded files
vac = pd.read_csv('/mnt/data/vacation_complete_dataset.csv')
vacmot_segmentation = pd.read_csv('/mnt/data/vacmot_segmentation_variables.csv')
vacmot_descriptor = pd.read_csv('/mnt/data/vacmotdescriptor_variables.csv')

# Inspect column names and dimensions
print('Vacation Data Column Names:', vac.columns.tolist())
print('Vacation Data Dimensions:', vac.shape)

print('Segmentation Variables Column Names:', vacmot_segmentation.columns.tolist())
print('Segmentation Variables Dimensions:', vacmot_segmentation.shape)

print('Descriptor Variables Column Names:', vacmot_descriptor.columns.tolist())
print('Descriptor Variables Dimensions:', vacmot_descriptor.shape)

# Summary statistics for specific columns in the vacation dataset
summary_cols = ['Gender', 'Age', 'Income', 'Income2']
print(vac[summary_cols].describe(include='all'))

# Visualize gender distribution
sns.countplot(x='Gender', data=vac)
plt.title('Gender Distribution')
plt.show()

# Visualize age distribution
sns.histplot(vac['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Visualize income distribution using Income2
sns.countplot(x='Income2', data=vac, order=vac['Income2'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Income2 Distribution')
plt.show()

# Check for missing values
missing_values = vac.isnull().sum()
print('Missing Values in Vacation Data:\n', missing_values[missing_values > 0])

# Replace missing Income with the median as an example
vac['Income'].fillna(vac['Income'].median(), inplace=True)
vac['Income2'].fillna('Unknown', inplace=True)
print('Data Cleaning Complete.')
