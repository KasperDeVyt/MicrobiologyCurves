Groeicurves.py creates LinEcoli.png.
It takes your growth data and plots OD in function of time (hours).
Then you can choose to zoom in on a certain part allowing you to plot the best fitting regression line.
This file is completely linear.

Automatic Best Regression.py and Manual Log regression.py do similar things and can both create LogEcoli.png
Automatic Best Regression.py has an automatic function that looks for the best regression based on R^2 value.
However that isn't always the wanted regression line hence Manual Log regression.py allows you to choose your regression line manually.
These plots are in Y-axis Log (ln2) and X-axis Linear

GrowthCurveAnnotation.py creates GroeicurveEcoli.png
It creates the growth curve you created earlier with a log axis and adds a predicted (purely illustrative) stationary and death phase.
This code also allows you to highlight the different phase of the bacterial growth.
