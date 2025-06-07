# This project analyzes recipes to predict ratings based on prep time and nutrition.
Author Junran Yang
## Overview and Introduction  
**DSC80 Final Project | UCSD**  

As a data science student and fitness enthusiast, I've always believed nutrition is the foundation of physical transformation. "You can't out-train a bad diet" – this project explores how data science can help us make better nutritional choices by analyzing over 80,000 recipes from Food.com.
Key Questions Explored:
How does protein content relate to recipe ratings?
Can we accurately predict calories from nutritional features?
Are there biases in how recipes are rated or described?
This work combines exploratory analysis, hypothesis testing, and predictive modeling to uncover insights for health-conscious cooks and data scientists alike.

I analyze two interconnected datasets from Food.com:

1. Recipes Dataset (83,782 recipes × 12 columns)

| Column             | Description                                                                                                                                                                                       |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `'name'`           | Recipe name                                                                                                                                                                                       |
| `'id'`             | Recipe ID                                                                                                                                                                                         |
| `'minutes'`        | Minutes to prepare recipe                                                                                                                                                                         |
| `'contributor_id'` | User ID who submitted this recipe                                                                                                                                                                 |
| `'submitted'`      | Date recipe was submitted                                                                                                                                                                         |
| `'tags'`           | Food.com tags for recipe                                                                                                                                                                          |
| `'nutrition'`      | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| `'n_steps'`        | Number of steps in recipe                                                                                                                                                                         |
| `'steps'`          | Text for recipe steps, in order                                                                                                                                                                   |
| `'description'`    | User-provided description                                                                                                                                                                         |
| `'ingredients'`    | Text for recipe ingredients                                                                                                                                                                       |
| `'n_ingredients'`  | Number of ingredients in recipe                                                                                                                                                                   |

Key Insight: The nutrition data uses Percentage Daily Values (PDV). I will add a column called protein_gram that converts protein to grams for interpretability

2. Interactions Dataset (731,927 reviews × 5 columns)

| Column        | Description         |
| :------------ | :------------------ |
| `'user_id'`   | User ID             |
| `'recipe_id'` | Recipe ID           |
| `'date'`      | Date of interaction |
| `'rating'`    | Rating given        |
| `'review'`    | Review text         |

## Data Cleaning and Exploratory Data Analysis
To prepare the recipe data for analysis, I performed several critical cleaning steps. First, I replaced zero values in the ratings column with NaN (using merged['rating'].replace(0, np.nan)), as these likely represent missing data rather than true zero-star ratings on Food.com's 1-5 scale. Next, I calculated average ratings per recipe by grouping by recipe ID and merging the results back into the dataset (groupby('id')['rating'].mean()), creating a consistent target variable for analysis. The nutrition data—originally stored as a string resembling a list (e.g., "[calories, fat, sugar,...]")—was split into separate columns (str.strip('[]').str.split(',').apply(pd.Series)) to enable individual analysis of each nutritional component. I then added a new column called `protein_grams` by stripping whitespace and converting values in `protein` to floats, and finally converted protein from Percentage Daily Value (PDV) to grams (protein * DAILY_PROTEIN_GRAMS / 100), making the values more interpretable for fitness applications. These steps collectively transformed raw data into an analysis-ready format by handling missing values, decomposing packed fields, standardizing units, and ensuring numeric consistency—all while preserving the dataset's integrity for downstream modeling and visualization.


| name                                 |     id |   minutes |   contributor_id | submitted   | tags                                                                                                                                                                                                                        | nutrition                                    |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                    |   n_ingredients |   user_id |   recipe_id | date       |   rating | review                                                                                                                                                                                                                                                                                                                                           |   average_rating |   calories |   total_fat |   sugar |   sodium |   protein |   saturated_fat |   carbohydrates |   protein_grams |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|----------:|------------:|:-----------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------:|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat , stirring frequently , until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs , sugar , cocoa powder , vanilla extract , espresso , and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean , about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                  | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour'] |               9 |    386585 |      333281 | 2008-11-19 |        4 | These were pretty good, but took forever to bake.  I would send it ended up being almost an hour!  Even then, the brownies stuck to the foil, and were on the overly moist side and not easy to cut.  They did taste quite rich, though!  Made for My 3 Chefs.                                                                                   |                4 |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 |             1.5 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl , sift together the flours and baking powder', 'set aside', 'in another mixing bowl , blend together the sugars , margarine , and salt until light and fluffy', 'add the eggs , water , and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop , scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !'] | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                    |              11 |    424680 |      453467 | 2012-01-26 |        5 | Originally I was gonna cut the recipe in half (just the 2 of us here), but then we had a park-wide yard sale, & I made the whole batch & used them as enticements for potential buyers ~ what the hey, a free cookie as delicious as these are, definitely works its magic! Will be making these again, for sure! Thanks for posting the recipe! |                5 |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 |             6.5 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce', 'pour into baking dish , sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly , about 10 more minutes']                                                                                                                                                                                                                                                                                                                              | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |     29782 |      306168 | 2008-12-31 |        5 | This was one of the best broccoli casseroles that I have ever made.  I made my own chicken soup for this recipe. I was a bit worried about the tsp of soy sauce but it gave the casserole the best flavor. YUM!                                                                                                                                  |                5 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |            11   |

### Univariate Analysis
For the univariate analysis, I created a histogram that visualizes the distribution of average recipe ratings in my dataset. The distribution shows that the vast majority of recipes have high average ratings, with a strong concentration near 5 stars. Very few recipes fall below a 4-star average, suggesting either user bias toward high ratings or a tendency to only review well-liked recipes.
<iframe
  src="assets/fig1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


### Bivariate Analysis
For the bivariate analysis, I created a scatter plot that visualizes the relationship between preperation time and average_rating. The scatter plot shows that recipes receive high average ratings are clustered at the 5-star mark across all time ranges, but mostly in the low preperation time range. One potention reason is that most data we had didn't have super long preperation time, and user really like to rate 5 star for the recipe, so we can't say there is a clear correlation between preparation time and average rating right now.
<iframe
  src="assets/fig2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Interesting Aggregates
For the aggregates analysis I created a table that identifies the top 10 recipe contributors by volume, revealing that the most prolific users (submitting 1,680-3,060 recipes each) consistently maintain exceptional ratings between 4.72-4.85 stars, with median prep times of 25-55 minutes. The results highlight both the platform's reliance on a small group of super-users for quality content and a potential positivity bias in ratings, as no top contributor averaged below 4.7 stars despite varying recipe volumes and preparation times.


|   recipe_count |   average_rating |   minutes |
|---------------:|-----------------:|----------:|
|           3060 |          4.78761 |        25 |
|           2754 |          4.84693 |        42 |
|           2503 |          4.80217 |        23 |
|           2436 |          4.78994 |        40 |
|           2368 |          4.76688 |        30 |
|           2310 |          4.82568 |        27 |
|           1867 |          4.81609 |        35 |
|           1795 |          4.77882 |        25 |
|           1711 |          4.71974 |        55 |
|           1680 |          4.85124 |        30 |




## Assessment of Missingness
### NMAR Analysis
The review column is likely NMAR (Not Missing At Random) because its missingness depends on the unobserved true sentiment of users—people may skip writing reviews when they feel ambivalent or dissatisfied, but this reason is not captured in the data. To make it MAR (Missing At Random), we would need additional data such as the time spent on recipe page or the clicks user made.

### Missingness Dependency
I investigated whether missing descriptions depend on recipe attributes by conducting two permutation tests:
Prep Time (minutes): Tested if recipes with missing descriptions have systematically different prep times.
Rating (rating): Tested if missing descriptions correlate with recipe ratings.

> Description and Prep Time (minutes)
Null Hypothesis (H₀): Description missingness is independent of prep time.
Alternative Hypothesis (H₁): Recipes with missing descriptions have different average prep times.
Test Statistic: Absolute difference in mean prep time (missing vs. non-missing).
Significance Level: α = 0.05
Result: p = 0.462 → Fail to reject H₀
Conclusion: No evidence that prep time affects description missingness.
<iframe
  src="assets/fig3.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
From the graph it can be seen that the distribution of prep time with description and without description look similar, meaning the missingness of description doesn't affect the overall shape of prep time distribution.

> Description and rating
Null Hypothesis (H₀): Description missingness is independent of rating.
Alternative Hypothesis (H₁): Recipes with missing descriptions have different average ratings.
Test Statistic: Absolute difference in mean ratings (missing vs. non-missing).
Significance Level: α = 0.05
Result: p = 0.014 → Reject H₀
Conclusion: Missing descriptions correlate with ratings (likely NMAR). Lower-rated recipes may lack descriptions because contributors invest less effort in unpopular recipes.
<iframe
  src="assets/fig4.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
From the graph it can be seen that the distribution of rating with description and without description look very different, meaning the missingness of description is indeed related to rating and could affect the shape of rating distribution.

## Hypothesis Testing
I am interested in whether recipes with different protein content tend to have different preparation times. Specifically, I want to know if high-protein recipes take longer to prepare than low-protein ones.

Null Hypothesis: High-protein and low-protein recipes take the same amount of time to prepare on average.
Alternative Hypothesis: High-protein recipes take longer to prepare than low-protein recipes.
Test Statistic: The difference in mean preparation time between high-protein and low-protein recipes.
Significance Level: 0.05

I chose a permutation test because I want to assess whether the observed difference in means could arise purely by chance under the null hypothesis. I split the recipes into two groups based on whether their protein content is above or below the median protein value. The observed difference in average preparation time was computed, and  then randomly shuffled the minutes column 500 times to simulate the distribution of mean differences under the null.

The resulting p-value was 0.26, meaning that 26% of the time, a difference as extreme or more extreme than the observed difference would occur by chance. Since this p-value is greater than our chosen significance level of 0.05, we fail to reject the null hypothesis. This suggests that there is not enough evidence to conclude that high-protein recipes take longer to prepare than low-protein ones. The test and design choices are appropriate for answering our question because they directly compare the means of two comparable groups under minimal assumptions, allowing us to assess if any observed difference is likely due to random variation.


## Framing a Prediction Problem
I plan to predict the calorie content of recipes using nutritional features, framing this as a regression problem since calories are continuous values. The response variable is calories—a critical metric for dietary planning—and I’ve chosen it because I often estimate meals based on partial ingredient data (e.g., sugar or protein amounts) rather than exact measurements.
To evaluate performance, I’ll use:
RMSE (Root Mean Squared Error): To quantify average prediction error in interpretable calorie units, prioritizing precise estimates for health tracking.
R² score: To measure how well the model explains calorie variance compared to a baseline.
I’ll use available nutritional features (sugar, protein, fat, etc.) and recipe attributes (minutes, n_steps) as predictors, since these are known before cooking—mirroring real-world scenarios where I might tweak ingredients. The model’s goal is to help make informed decisions even when exact measurements are unavailable.

## Baseline Model
My baseline uses a simple linear regression with two quantitative features: protein (grams) and sugar (grams), both standardized via StandardScaler. No encoding was needed since these are numeric values. The model achieves:

Train RMSE: 763.64
Test RMSE: 315.35 calories
R²: 0.73 on test data, 0.72 on train data.

While the R² indicates a moderately strong relationship, the large train-test RMSE gap raises concerns about overfitting or split randomness. My model’s strength lies in its simplicity—using only two intuitive features, but future iterations should incorporate more nutritional variables (e.g., fat, carbs) to improve accuracy.

## Final Model
Final Model Features
'has_goodtag'
This binary feature identifies recipes likely to be desserts/sweets by checking reviews for terms like 'sugar' or 'dessert'. We included it because desserts typically have distinct calorie profiles due to higher sugar/fat content. The feature was one-hot encoded to avoid imposing ordinal relationships.

'n_steps' (binarized)
I transformed this into a binary feature (≥9 steps = complex) because intricate recipes often use more calorie-dense ingredients (e.g., sauces, layered components). This aligns with culinary intuition that complex preparations tend to be richer.

Nutritional features ('protein', 'sugar', etc.)
I included five standardized nutritional components (protein, sugar, sodium, saturated_fat, carbohydrates) because calories derive directly from macronutrients (4 cal/g for protein/carbs, 9 cal/g for fat). Polynomial features (degree=1) were tested but not used, as cross-validation showed higher degrees increased overfitting without improving RMSE.

Modeling and Hyperparameter Tuning
I used linear regression for its interpretability and alignment with nutritional science (calorie calculation is inherently linear). The key hyperparameter—polynomial degree—was selected via 5-fold cross-validation (minimizing RMSE), with degree=1 performing best. This suggests calorie predictions rely primarily on additive nutrient contributions rather than complex interactions.


Using linear regression with polynomial features (degree=1), selected via 5-fold cross-validation (minimizing RMSE), the model achieves:
Test RMSE: 166.06 calories (47% lower than baseline)
Test R²: 0.926 (27% improvement over baseline)

The near-identical train/test scores (RMSE: 157.16 vs 166.06, R²: 0.926 vs 0.926) suggest much better generalization.

## Fairness Analysis
For the fairness evaluation, I split recipes into two groups:
Group X (Meat recipes): Recipes containing meat, identified by tags like 'beef', 'chicken', or 'fish'.
Group Y (Non-meat recipes): All other recipes.

I evaluated RMSE parity between these groups because calorie prediction errors could mislead dietary choices—for example, underestimating calories in meat dishes might encourage overconsumption, while overestimating vegetarian recipes could unnecessarily deter healthier options.

Hypotheses:
Null (H₀): The model is fair. Its RMSE for meat and non-meat recipes is equal, and any differences are due to random chance.
Alternative (H₁): The model is unfair. Its RMSE differs between meat and non-meat recipes.

Test Design:
Test Statistic: Absolute difference in RMSE.
Significance Level: 0.05.
Permutation Tests: 500 shuffles of the has_meat labels to simulate H₀.

Results:

Observed Difference: 17.023214562489272 calories.

p-value: 0.262 (> 0.05).

Conclusion:
We fail to reject H₀ (p > 0.05), finding no statistically significant evidence of unfairness. The model’s calorie predictions are similarly accurate for meat and non-meat recipes, with RMSE differences likely due to chance.
