# This project analyzes recipes to predict ratings based on prep time and nutrition.

## Project Overview  
**DSC80 Final Project | UCSD**  

As a data science student and fitness enthusiast, I've always believed nutrition is the foundation of physical transformation. "You can't out-train a bad diet" – this project explores how data science can help us make better nutritional choices by analyzing over 80,000 recipes from Food.com.
Key Questions Explored:
How does protein content relate to recipe ratings?
Can we accurately predict calories from nutritional features?
Are there biases in how recipes are rated or described?
This work combines exploratory analysis, hypothesis testing, and predictive modeling to uncover insights for health-conscious cooks and data scientists alike.

We analyze two interconnected datasets from Food.com:

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
