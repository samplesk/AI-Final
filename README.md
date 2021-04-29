# AI-Final
The purpose of this study is to learn how to use machine learning models on multiple datasets and to learn which models perform the best across several different metrics, including F1 score, accuracy, and hinge loss.
Accuracy: The level of correctness the model predicts, compared with what the actual data.
F1: Conveys the balance between the precision and the recall, an additional test of accuracy
Hinge Loss: another measure of accuracy, but scaled with 0 as the best.
We have chosen three different datasets to run through three different models. This should give an insight into not only which models are best, but what data types work we will them.

Results:
The classification model we chose was a k nearest neighbors model. We used 25 nearest neighbors and were biased by their distance. We also learned from 70% of the data sets and predicted on the remaining 30%. The model did the best on the zoo data, which was expected, with an 87% accuracy rate. The model did only slightly worse on the real estate valuation data at around 73% accuracy. The model struggled with the student performance data with a 36% and 32% accuracy on each half.

Regression Model: Linear

Dimensionality Reduction Model: Linear Discriminant Analysis

Data Sets:
  1. Zoo data
    -Best for classification
    -16 boolean valued attributes
    -2 numeric attributes
    -101 unique animals classified into 7 classes
  2. Student performance data
    -This data set has 33 features and 649 entries
    -The data is mostly categorical with a few numerical values.
  3. Real Estate
    -It is the market historical data set of real estate valuation.
