# Interview Preparation for Inventory Stockout Risk Prediction

## 1. Explain your project.
This project predicts whether a product is likely to run out of stock in the next 7 days. I used sales, inventory, lead time, and promotion data to build a classification model. The goal was to identify high-risk products early so the business can reorder on time.

## 2. Why did you choose this project?
I chose this project because it is realistic, business-focused, and easy to explain in interviews. It connects data science with a common supply-chain problem and shows skills in EDA, feature engineering, and machine learning.

## 3. What is stockout?
A stockout happens when a product has no inventory left or not enough stock to meet demand before the next supply arrives.

## 4. Why is stockout prediction a classification problem?
Because the output is a yes or no answer. We predict whether the product is likely to stock out in the next 7 days or not. That is a binary classification problem.

## 5. What was your target variable?
My target variable was stockout_risk. It was 0 for no stockout risk and 1 for high stockout risk within the next 7 days.

## 6. How did you create the target?
I used a business rule based on expected future demand and available stock. If expected demand for the next 7 days was greater than current stock plus incoming stock, or if stock was too low relative to lead time demand, it was marked as risky.

## 7. What features did you use?
I used current stock, daily sales, reorder level, lead time, incoming stock, promotion, price, holiday, returns, and derived features like days_of_inventory, lead_time_demand, stock_gap, and sales_growth.

## 8. How did you perform EDA?
I checked data quality, missing values, duplicate rows, category distribution, sales distributions, stock ranges, stockout rate by category, and relationships like sales versus stock and lead time versus stockout risk. I used histograms, bar plots, box plots, and heatmaps.

## 9. What feature engineering did you perform?
I created features such as average_daily_sales, sales_7d_avg, sales_30d_avg, days_of_inventory, lead_time_demand, stock_gap, reorder_gap, promotion_flag, and weekend_flag. These features explain inventory pressure and replenishment risk.

## 10. Why did you use Logistic Regression?
It is a strong baseline model for binary classification. It is simple, fast, easy to explain, and works well as a starting point.

## 11. Why did you use Random Forest?
Random Forest handles nonlinear relationships well and gives better performance than a simple baseline in many business datasets. It is also easy to explain through feature importance.

## 12. Why is Recall important?
Recall tells us how many actual risky products we caught. In stockout prediction, missing a risky product is expensive because it can lead to lost sales and poor customer experience.

## 13. What is a false negative in this project?
A false negative is when the model predicts no risk, but the product actually goes out of stock. This is costly because the business may not replenish in time.

## 14. What is data leakage?
Data leakage is when information from the future is used while training the model. For example, using next-week sales or next-week stock levels as model inputs would make the model look better than it really is.

## 15. How did you split the data?
I used a chronological split. Training data was from earlier dates, while validation and test data were from later dates. This is more realistic for time-based inventory data and reduces leakage.

## 16. How did you handle missing values?
I checked missing values first, then filled numeric missing values with median values and filled categorical missing values with the most common category.

## 17. How did you handle categorical variables?
I used OneHotEncoder inside a preprocessing pipeline. This converts category labels into numeric values that the model can process.

## 18. How did you evaluate the model?
I used accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC. I also looked at business impact, not only accuracy.

## 19. Which model performed best and why?
The best model was the one that balanced recall and precision for the business objective. In this project, the model with stronger recall and stable ROC-AUC was preferred because it caught more risky products.

## 20. What business value does the project provide?
The project helps businesses act earlier on products that may run out of stock. It supports quicker replenishment decisions and reduces lost sales.

## 21. What are the limitations?
The data is synthetic, so it may not fully match real business conditions. Unexpected demand changes, supplier delays, and external events can still affect performance.

## 22. What would you improve in the future?
I would add real inventory and ERP data, better demand forecasting, supplier reliability features, and model retraining. I would also monitor drift over time.

## 23. How would you deploy the model?
I would package the training and prediction logic, save the model, and expose it through a simple API or Streamlit dashboard for business users.

## 24. What was your contribution as a Data Scientist?
My main contribution was understanding the business problem, cleaning the data, performing EDA, engineering useful features, training and evaluating models, and deriving business insights that help support replenishment decisions.
