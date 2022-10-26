# Linear Regression

This sub-repository demonstrates the implementation of Linear Regression algorithm on the time series financial data.

Contents of **Linear Regression**

* [Image](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/05/2.3.png): contains images used in README
* [Linear_Reression.ipynb](): Jupyter notebook file contains
  * a. Introduction of Linear Regression
  * b. Building Linear Regression algorithm from scratch and implement Linear Regression in two different time series data (Ethereum price and Bitcoin price), and predict the future prices
  * c.  Feature Engineering

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/Linear%20Regression/Image/linear-regression.png)

### A Short Summary

# Linear Regression

[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) is a classic supervised model that dates back over 200 years. Linear regression combines a specific set of input values (x) with a predicted output (y). Both input and output variables are numeric. The linear equation assigns one scale factor to each input value (often denoted "m"), and a constant (often denoted "b") that gives the line an additional degree of freedom because it can move up and down on a 2D plane. Thus, we have the familiar equation from algebra: **y = mx + b** (below). In the image above, the red line is referred to as the **line of best fit** described by this equation.

![image](https://github.com/ppunia74/INDE-577_Fall2022/blob/main/SupervisedLearning/Linear%20Regression/Image/regression-equation.jpeg)

In higher dimensions, the line is called a plane or "hyperplane", and simply adds additional terms to the equation, as written below:

$$y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n$$


When the coefficient of one of these terms becomes zero, it effectively removes the influence of the input variable on the model.

Some of the commonly used linear regression based methods::
- **Simple linear regression:** The model has a single input variable and uses statistics to estimate coefficients. It requires computing statistical properties like means, standard deviations, correlations, and variance. 
- **Ordinary least squares** ("or ordinary least squares linear regression" or "least squares regression"): This is the widely used. When there is more than one input, OSLR can estimate the values of the coefficients. The [Gauss-Markove theorem](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) ensures this method minimizes the "sum of the squared residuals." Essentially a regression line through the data, calculate the distance from each data point to the regression point, square it, and sum all the squared errors together. This method uses the matrix and linear algebra to estimate the optimal values for the coefficients. It requires memory to fit the data and perform matrix operations but is very fast to calculate.
- **Gradient Descent:** This approach is further discussed [elsewhere](https://github.com/ppunia74/INDE-577/blob/main/supervised%20learning/1%20-%20gradient%20descent/README.md) in this repository, but in short this method starts with random values for the coefficients, then iteratively works to minimize the error of the model.
- **Regularization:** These models seek to minimize the sum of the squared error of the model on training data using ordinary least squares, as well as reducing the complexity of the model. These methods are effective when input values have collinearity in input values, and ordinary least squares would overfit the training data.
  - [LASSO Regression](https://en.wikipedia.org/wiki/Lasso_(statistics)) (aka Least Absolute Shrinkage and Selection Operator, or L1 regularization): modifies ordinary least squares to also minimize the absolute sum of the coefficients by pulling data values towards the mean. This algorithm is useful in feature selection and applies a "penalty" to features that do not benefit the prediction
  - [Ridge Regression](https://en.wikipedia.org/wiki/Tikhonov_regularization) (aka L2 Regularization): modifies ordinary least squares to minimize the squared absolute sum of the coefficients and compensate for large variation in input data. It includes a penalty term (often denoted lambda) that shrinks the weight of probability theta
  - [Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization): is a combination of LASSO and Ridge Regression that adjusts both the weight of the features and probability

Linear models have following assumptions:
- Linearity: the dependent variable Y should be linearly related to independent variables - view using a scatter plot

![image](https://editor.analyticsvidhya.com/uploads/96503linear-nonlinear-relationships.png)

- Normality: X and Y values follow the normal distribution - check with histograms, KDE plots, or Q-Q plots

![image](https://editor.analyticsvidhya.com/uploads/64526normality.png)

- Homoscedasticity: The spread of the residuals should be constant for all variables - use a residual plot (if the assumption is violated, points will form a funnel shape)

![image](https://editor.analyticsvidhya.com/uploads/51367residuals.png)

- Independence/No multicollinearity: there is no correlation between any of the independent variables - calculate a correlation matrix of VIF score (if VIF > 5, variables are highly correlated)

![image](https://editor.analyticsvidhya.com/uploads/99214correlation.png)

- Error terms are also normally distributed: plot histograms and Q-Q plots

![image](https://editor.analyticsvidhya.com/uploads/79532normality%20of%20error.png)

- No autocorrelation: error terms should be independent of each other - use Durbin-Watson test, where the null hypothesis assumes there is no autocorrelation

![image](https://editor.analyticsvidhya.com/uploads/38946DW.png)

There are also a few evaluation metrics for regression:
- $R^2$, aka the "coefficient of determination": the most common metric, is the ratio of variation to the total variation (equation below - SS_res is the residual sum of squares and SS_tot is the total sum of squares). The value will be between 0 and 1; the closer to 1, the better the model. However, as the number of features increases, the value of $R^2$ increases, giving the (sometimes false) illusion of a good model.

![image](https://editor.analyticsvidhya.com/uploads/74264r2.png)

- Adjusted $R^2$: improvement to $R^2$, only considers features important for the model and shows the real improvement. The equation is detailed below

![image](https://editor.analyticsvidhya.com/uploads/80741adjusted%20r2.png)

- Mean Squared Error (MSE)

![image](https://editor.analyticsvidhya.com/uploads/42113mse.jpg)

- Root Mean Squared Error (RMSE): root of the mean difference between actual and predicted values. It penalizes large errors

![image](https://editor.analyticsvidhya.com/uploads/69457rmse.png)


The above information was largely based on [this article](https://www.analyticsvidhya.com/blog/2021/05/all-you-need-to-know-about-your-first-machine-learning-model-linear-regression/), which can be referenced for further reading.


In machine learning, the objective of using linear regression is used to find the mathematical equation that best explains the relationship between the response variable (<img src="https://latex.codecogs.com/svg.image?\mathbf{Y}" title="\mathbf{Y}" />) and the predictors (<img src="https://latex.codecogs.com/svg.image?\mathbf{X}" title="\mathbf{X}" />). The matrix notation of the linear model is 

<img src="https://latex.codecogs.com/svg.image?\mathbf{Y}=\mathbf{X}\mathbf{\beta}&plus;\mathbf{\epsilon}" title="\mathbf{Y}=\mathbf{X}\mathbf{\beta}+\mathbf{\epsilon}" />, where <img src="https://latex.codecogs.com/svg.image?\beta" title="\beta" /> is unknown model parameter, and <img src="https://latex.codecogs.com/svg.image?\epsilon" title="\epsilon" /> is random error.

### Method of Least Squares

The model fitting with Least Squares aims to minimize the sum of squared errors (SSE). The estimate of <img src="https://latex.codecogs.com/svg.image?\mathbf{Y}" title="\mathbf{Y}" /> by this model is <img src="https://latex.codecogs.com/svg.image?\hat{Y}=&space;\mathbf{X}\hat{\beta}=\mathbf{X}\left(&space;\mathbf{X}^{T}\mathbf{X}&space;\right)^{-1}\mathbf{X}^{T}\mathbf{Y}" title="\hat{Y}= \mathbf{X}\hat{\beta}=\mathbf{X}\left( \mathbf{X}^{T}\mathbf{X} \right)^{-1}\mathbf{X}^{T}\mathbf{Y}" />.

The hat matrix is defined as <img src="https://latex.codecogs.com/svg.image?H=\mathbf{X}\left(&space;\mathbf{X}^{T}\mathbf{X}&space;\right)^{-1}\mathbf{X}^{T}" title="H=\mathbf{X}\left( \mathbf{X}^{T}\mathbf{X} \right)^{-1}\mathbf{X}^{T}" />, 
which is composed solely of the sample values of the predictor variables.

Please read [Linear_Reression.ipynb](https://github.com/cissyyang1014/DataScience_and_MachineLearning/blob/main/SupervisedLearning/Linear%20Regression/Linear_Regression.ipynb) to learn more details.

---

### Datasets

