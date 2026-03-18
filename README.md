## 📊 Bank Marketing Campaign Prediction

This project analyzes the "Bank Marketing Dataset", which contains data from direct marketing campaigns conducted by a Portuguese banking institution via telephone calls. The objective of these campaigns was to promote subscriptions to a bank term deposit product (target variable: **`y`**).

🎯 Objective

To build predictive models that identify customers who are most likely to subscribe to a term deposit, enabling more efficient and targeted marketing strategies.

⚙️ Methods

I implemented two ensemble machine learning models:

**Random Forest Classifier**
* **XGBoost Classifier**

These models were selected for their strong performance on structured/tabular data and their ability to capture nonlinear relationships.

📏 Evaluation Strategy

Model performance was evaluated using logarithmic loss (log loss) to ensure that predicted probabilities were well-calibrated and not overconfident guesses. This metric penalizes incorrect predictions more heavily when the model is highly confident but wrong.

📈 Probability-Based Ranking

Instead of using only binary predictions, customers were ranked by their predicted subscription probabilities. This enables:

* Prioritized outreach to high-likelihood customers
* Better allocation of marketing resources
* More realistic business decision-making

✅ Outcome

The resulting models provide a data-driven approach for improving campaign efficiency through probability-based customer targeting.

