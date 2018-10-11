# Problem Definition
The system is built to judge the quality of wine to be good or bad. By inputting basic attributes of wine like density, pH value, etc. , the system predicts the quality of the given alcohol. We define the task as a binary classification problem. With a huge amount of training data on various wine attributes, we are able to train a classifier for predictions. The system implements Random Forest and Support Vector Machine algorithm and parameters are going to be updated in the training process, in order that our model gradually fits the training data. A well trained classifier is able to predict whether an alcohol is of good quality.

# Background
As a luxury good, wine is enjoyed by consumers all around the world. During the process of producing wine, certification and quality assessment are often included, as these steps assure the quality of wine before transported to the market. Such tasks are usually done by human experts, which means a lot of human resources are spent on them. However, with the increase of wine consumption nowadays, more efficient methods are necessary to assess the quality of wine.
Certification is generally implemented by using physicochemical and sensory tests. From the test, a series of data tested by sensors such as pH, alcohol and chlorides are significant factors for quality evaluation. Based on these factors, multiple techniques could be applied to evaluate the quality of wine[2]. 

In general, the challenge for wine judging system is that physicochemical laboratory tests routinely used to characterize wine, and Sensory tests rely mainly on human experts. Therefore, there is a lack of agreement on the quality term in general. During the process of developing our system, there are two main challenges we faced. First one is the imbalance problem. As the result showed in the Model using SVM with Linear kernel, the recall is ridiculously low due to the class imbalance. Secondly, our training time is too long that hindered our training efficiency. 

In our work, we implement classification algorithms to estimate the quality of wine. One is Support Vector Machine(SVM), and the other is Random Forest. We compare the prediction performances of these two different algorithms and choose our proper models using cross-validation.

# Data Description
The dataset is downloaded from UCI Machine Learning Repository[1]. The dataset includes two subsets, related to red and white vinho verde wine samples, from the north of Portugal. There are 4898 instances, one instance composes 11 attributes which are separately fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol. The data label is the quality score represented as an integer from 0 (very bad) to 10 (very excellent). The score is objectively tested by wine experts.

# Data Preprocessing 
As mentioned above, since we define the task as a binary classification problem, we pre-processed and binarized the labels of data. We labeled 0(bad) for all the wines that scored below 7 and 1(good) for the rest of wines that have scores higher or equal to 7. The ratio of good wine is 0.22 while the ratio of bad wine is 0.78 in the dataset we use. 

# Feature selection and Dimensionality reduction
In order to improve the training efficiency, we performed feature selection using Random Forest and the 10 most important features are selected. The feature “alcohol” is the most important while the feature “fixed acidity” is the least important and is thus deleted. Although there are only 11 attributes in our original dataset, over-sifting will lead to the decrease of accuracy. The application of LDA allowed us to reduce the dimensionality of dataset to 2D or 3D where we are able to visualize the dataset below. 

# Experimental Design
We implement Over-sampling minority class using SMOTE(Synthetic Minority Oversampling technique) in order to solve the problem of imbalancing in our data. Also, we perform feature selection using Random Forest to choose the 10 most important features. Next, we use double-resampling for model selection, and pick up models with best performances for the wine quality prediction task. Lastly, we compare the performances of models by analysing confusion metrics based on accuracy, specificity and F1 score.

We use two classification algorithms: Random Forest[4] and Support Vector Machine[3] with 3 different kernels(RBF, Poly and Linear). Random Forest performs well in classification problem by deploying multiple decision trees which will improve the accuracy. SVM is a popular classifier used to draw a plane among multiple attributes, thus it will be efficient in this dataset. 

In the model selection step, we use double-resampling with 5-fold cross validation and Leave-one-out cross validation. By the implementation of both exhaustive and non-exhaustive method, we are able to select proper kernels and parameters in SVM and number of trees in Random Forest.
Confusion Matrix is used for the comparison among different models, and we try to compare the different models by analysing metrics including Accuracy, Precision, Recall, Specificity and F-Value. 

# Results
Classification results of SVM and Random Forest on our validation set are listed below:

Model                     Accuracy Precision Recall Specificity F1 score

Support Vector Machine       85%        98%     30%      65%        0.46

Random Forest                88%        79%     57%      76%        0.66


# Software and hardware
This project will basically developed using Jupyter Notebook on a computer with a 2.8 GHz Intel Core i7 CPU and a 8GB RAM. The programming language we chose to use is Python and the library we imported including numpy, pandas, sklearn and smote.

# Conclusion
Based on our results, we conclude that both Support Vector Machine and Random Forest are good at solving the binary wine quality classification problem since the accuracy of both models are considerably high and close. In order to perform the judgement more accurate, a possible way is to do regression rather than classification according to the attributes.

# Reference
[1] Asuncion, Arthur, and David Newman. "UCI machine learning repository." (2007).
[2] S. Charters, S. Pettigrew. The dimensions of wine quality. Food Quality and Preference, 18 (2007), pp. 997-1007
[3] Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." Machine learning 20.3 (1995): 273-297.
[4] Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.

