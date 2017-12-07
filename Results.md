---
nav_include: 6
title: Conclusion
notebook: Final_Project.ipynb
---


We predicted the murder rate (murders per 100,000 inhabitants) in US MSAs using several models and predictors that captured information on demographics, employment, income, education, etc.  The best performing model was Random Forest with an R-squared of 0.54 and Mean Absolute Error (MAE) of 1.6.  To estimate and score the model, we split the data based on specific years (for example, years 2006-2015 data was used to train the model while 2016 data was used to test the model).  Visualizations as well as the code used to perform this analysis can be seen in the Models page.

We created a second dataset that included employment levels and wages for industries that seem to be related to crime, such as police and mental health counselors. The R-squared of this model improved by 0.02 while the MAE improved by 0.01.

There are a couple of shortcomings that we would have liked to remedy given more time and/or data.  First, it could be more informative if we had city or neighborhood level data as this would capture heterogeneity within cities.  Also, information on gun ownership, drug use or gang prevalence, and amount of jail time (i.e. state laws and changes to federal laws) could have led to a more accurate model, as we'd expect these variables to be highly correlated with crime.
