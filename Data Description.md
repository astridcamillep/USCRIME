---
nav_include: 1
title: Data Description
---

## Contents
{:.no_toc}
*  
{: toc}

Here's a brief description of the different sources of data used and the procedures applied for its analysis. 

## Sources of Data

### FBI

Data on crime was scraped directly from the FBI website. Here we were able to find information on different types of crime, such as murder, violent crime, rape, thefts, robbery and others, for all Metropolitan Statistical Areas (MSAs) in the US. This database covers the period 2006-2016. 

We used the murder rate (murders divided by population) as the outcome variable in our model.

### United States Census Bureau

Data on demographic variables was collected through the United States Census Bureau. Our compiled database includes over 300 variables at the MSA level. 

Among the variables included there's detailed information:
1. Poverty levels by age group, race and gender
2. Employment status by age group, race and gender
3. Number of female individuals by age group
5. Median earnings by age, race, gender and educational attainment
5. Marriage status
6. Mean household income and income per capita by race
5. Population enrolled in either a public or private academic institution
6. Housing costs
5. Educational attainment
6. Sex ratio
7. Unemployment rate by age group

This database covers the period 2006-1016. 

### Bureau of Labor Statistics (BLS)

As an additional source of data, we use the total number and average hourly wage of police and sheriff?s patrol officers and mental health counselors by MSA.  Here we again have data for our period of interest (2006-2016).

This data was used in a secondary model as a significant number of MSAs didn?t report one or more of the BLS indicators.

## Data Cleaning and Merging

To handle missing values in the Census and BLS data, we use a mean-imputation approach. For example, if a given variable is missing a value for a year, we use its historical mean as a substitute. 

For merging our different datasets we use the variables year and MSAID, the latter is unique across MSAs and year and is common across our databases. 

