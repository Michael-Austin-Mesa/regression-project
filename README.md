# Regression Project: Single Family Property Values
# Project Description
Zillow wants insight on drivers of property value to reach actionable solutions.

# Project Goal
- Explore the effects of the number of bedrooms, bathrooms and square footage of Single Family Properties that had a transaction in 2017.
- Construct a ML regression model that accurately predicts property tax value.

# Initial Thoughts
My initial hypothesis is that drivers of tax value will likely be main features of most home and how many/much there are such as the number of bedrooms, bathrooms, and square footage.

# Plan

- Acquire data from zillow database in SQL

- Prepare data by dropping unnecessary columns, removing nulls, renaming columns, and optimizing data types.

- Explore data in search of drivers of property value and answer the following:

> Are the number of bedrooms related to tax value?


> Are the number of bathrooms related to tax value?


> Is square footage related to tax value?


> Is there an equal distribution of properties in each county?

- Develop a model to predict property value

> Use drivers identified in explore to build predictive models

> Evaluate models on train and validate data

> Select best model based on highest accuracy

> Evaluation of best model on test data

- Draw conlcusions

# Data Dictionary

| Feature | Definition |
| :- | :- |
| bedrooms | Integer, # of bedrooms in a property |
| bathrooms | Decimal value, # of bathrooms in a property, including fractional bathrooms |
| sq_feet | Integer, calculated total living area in a property |
| tax_value | Integer, total tax assessed value of the parcel, our target variable |
| year_built | Integer, the year a property was built |
| tax_amount | Decimal value, total property tax assessed for that assessment year |
| fips | Integer, Federal Information Processing Standard code (county) |

# Steps to Reproduce
1. Clone this repo
2. Acquire the data from SQL database
3. Place data in file containing the cloned repo
4. Run notebook

# Takewaways and Conclusions

- There is a positive correlation between the number of bedrooms and tax value of a property.


- There is a positive correlation between the number of bathrooms and tax value of a property.


- There is a positive correlation between the amount of square footage and tax value of a property.


- There is an uneven distribution of properties in the Los Angeles County, Orange County, an Ventura County.

# Recommendations

- Continue to focus on the number of bedrooms, bathrooms, and square footage of homes as drivers of tax value.


- Evaluate the tax value of properties when properties are separated into their respective counties.


- Create a model for each county that had properties with a transaction.

- Tableau Workbook for Zillow.csv: https://public.tableau.com/app/profile/michael.mesa/viz/ZillowDataStory/Story1
