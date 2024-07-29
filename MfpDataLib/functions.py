import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
basepath = Path(__file__).parent.parent

def process_myfitnesspal():
	(basepath/"data/myfitnesspal").mkdir(exist_ok=True, parents=True)
	(basepath/"data/myfitnesspal/processed").mkdir(exist_ok=True)

	entries = []
	with open(basepath/"data/myfitnesspal/raw/mfp-diaries.tsv", "r") as f:
		for line in f:
			line = line.strip()
			if not line:
				break

			parts = line.split("\t")
			user_id, date, meals, nutrients = parts

			meals = json.loads(meals)
			foods = []
			for meal in meals:
				meal_id = {"meal_name": meal["meal"], "meal_idx": meal["sequence"]}
				m_foods = meal["dishes"]
				m_foods = [meal_id | process_food(food) for food in m_foods]
				foods.extend(m_foods)

			nutrients = json.loads(nutrients)
			goal = nutrients["goal"]
			nutrients = nutrients["total"]

			goal = {x["name"].lower(): x["value"] for x in goal}
			nutrients = {x["name"].lower(): x["value"] for x in nutrients}

			entries.append({
				"user_id": user_id,
				"date": pd.to_datetime(date, format='%Y-%m-%d'),
				"foods": foods,
				"nutrients": nutrients,
				"goal": goal,
			})

	food_rows = []
	for entry in entries:
		for food in entry["foods"]:
			row = {
				"user_id": entry["user_id"],
				"date": pd.to_datetime(entry["date"], format='%Y-%m-%d'),
				**food,
			}
			food_rows.append(row)
	df = pd.DataFrame(food_rows)
	df.to_csv(basepath/"data/myfitnesspal/processed/myfitnesspal_meals.csv", index=False)

	user_rows = []
	for entry in entries:
		goal = {f"goal_{k}": v for k, v in entry["goal"].items()}
		row = {
			"user_id": entry["user_id"],
			"date": pd.to_datetime(entry["date"], format='%Y-%m-%d'),
			**entry["nutrients"],
			**goal,
		}
		user_rows.append(row)
	df = pd.DataFrame(user_rows)
	df.to_csv(basepath/"data/myfitnesspal/processed/myfitnesspal_goals.csv", index=False)

def process_food(food):
	output = {x["name"].lower(): int(x["value"].replace(",", "")) for x in food["nutritions"]}

	brand, food_name, flavor, serving_size = parse_food_name(food["name"])
	output["description"] = food["name"]
	output["food_name"] = food_name
	output["brand"] = brand
	output["flavor"] = flavor
	output["serving_size"] = serving_size

	return output

def parse_food_name(name):
	parts = name.split(", ")
	if len(parts) == 1:
		serving_size = None
	else:
		serving_size = parts[-1]
		name = ", ".join(parts[:-1])

	parts = name.split(" - ")
	if len(parts) == 1:
		return None, parts[0], None, serving_size
	elif len(parts) == 2:
		return parts[0], parts[1], None, serving_size
	else:
		return parts[0], parts[1], parts[2], serving_size
	
# Adjusted from Natalia's code:
def calc_user_stats(user_id, goals_df, meals_df, stats_df):

    user_goals = goals_df[goals_df['user_id'] == user_id].copy()

    # Convert the 'date' column to datetime
    user_goals['date'] = pd.to_datetime(user_goals['date'], format='%Y-%m-%d')

    # Initialize variables to store calculated statistics
    start_date = user_goals['date'].min()
    end_date = user_goals['date'].max()
    unique_dates = user_goals['date'].nunique()
    days_active = unique_dates

    # Calculate the differences between consecutive dates
    date_diffs = (user_goals['date'] - user_goals['date'].shift()).dropna()

    # Calculate average days between entry
    avg_days_between_entry = np.round(date_diffs.mean().days) if not date_diffs.empty else 0
    std_days_between_entry = np.round(date_diffs.std().days) if not date_diffs.empty else 0
    median_days_between_entry = np.round(date_diffs.median().days) if not date_diffs.empty else 0

    # Calculate longest hiatus
    longest_hiatus = np.round(date_diffs.max().days) if not date_diffs.empty else 0

    # Filter data for the current person_id
    person_meals = meals_df[meals_df['user_id'] == user_id]

    # Calculate average meals a day
    avg_meals_per_day = person_meals.groupby('date')['meal_idx'].nunique().mean()
    std_meals_per_day = person_meals.groupby('date')['meal_idx'].nunique().std()
    median_meals_per_day=  person_meals.groupby('date')['meal_idx'].nunique().median()

    # Number of Days Logged
    days_logged = int((end_date - start_date).total_seconds()/(3600*24)) + 1
    days_ratio = days_active/days_logged

    # Calorie goals:
    avg_calorie_goal = round((user_goals['goal_calories'].mean()))
    std_calorie_goal = round((user_goals['goal_calories'].std(skipna=True))) if len(user_goals['goal_calories'].dropna().unique()) > 1 else 0
    median_calorie_goal = round((user_goals['goal_calories'].median()))
    avg_calories_consumed = round((user_goals['calories'].mean()))
    std_calories_consumed = round((user_goals['calories'].std(skipna=True))) if len(user_goals['calories'].dropna().unique()) > 1 else 0
    median_calories_consumed =  round((user_goals['calories'].median()))


    # Append the calculated statistics to the stats_df
    stats_df = pd.concat([stats_df, pd.DataFrame({
        'Person ID': [user_id],
        'Start Date': [start_date],
        'End Date': [end_date],
        'Days Active': [days_active],
        'Days active to span Ratio':[days_ratio],
        'Days Logged': [days_logged],
        'Average Days Between Entry': [avg_days_between_entry],
        'Median Days Between Entry': [median_days_between_entry],
        'Standard Deviation Days Betweeen Entry': [std_days_between_entry],
        'Longest Hiatus (Days)': [longest_hiatus],
        'Average Meals Per Day': [avg_meals_per_day],
        'Median Meals Per Day': [median_meals_per_day],
        'Standard Deviation Meals': [std_meals_per_day],
        'Average Calorie Goal': [avg_calorie_goal],
        'Standard Deviation Calorie Goal': [std_calorie_goal],
        'Median Calorie Goal': [median_calorie_goal],
        'Average Calories Consumed per Day': [avg_calories_consumed],
        'Standard Deviation Calories Consumed': [std_calories_consumed],
        'Median Calories Consumed per Day': [median_calories_consumed]
    })], ignore_index=True)

    # Changes in Calorie Goal Graphs 
    plt.figure()
    plt.plot(user_goals['date'], user_goals['goal_calories'], marker='o', linestyle='-')  # Plot calorie goals over dates
    plt.scatter(user_goals['date'], user_goals['calories'], marker='o', color='orange')
    plt.axhline(user_goals['goal_calories'].mean(), color='green', linestyle='dashed', linewidth=1, label= 'Goal Calories Mean')
    plt.axhline(user_goals['goal_calories'].median(), color='red', linestyle='dashed', linewidth=1, label= 'Goal Calories Median')
    plt.title(f'Changes in Caloric Goals Person {user_id}')  
    plt.xlabel('Date')  
    plt.ylabel('Calorie Goal')  
    plt.xticks(rotation=45)  
    plt.tight_layout() 
    ax = plt.gca()
    ax.set_ylim([1300, 1800])
    plt.legend() 
    plt.show() 

    return stats_df

def init_user(userID, userGoals, userMeals, print_stats=False):

    (basepath/''.join(["data/userAnalysis/user",str(userID)])).mkdir(exist_ok=True, parents=True)

    if print_stats==True:
        print(len(userGoals), "daily entries with", len(userMeals["food name"].unique()), "total unique food items")

    diet_df = pd.DataFrame({'Calories': userGoals.calories/userGoals.goal_calories, 'Sodium': userGoals.sodium/userGoals.goal_sodium,
                            'Carbs': userGoals.carbs/userGoals.goal_carbs, 'Fat': userGoals.fat/userGoals.goal_fat, 
                            'Protein': userGoals.protein/userGoals.goal_protein,  'Sugar': userGoals.sugar/userGoals.goal_sugar})
    ax = sns.boxplot(diet_df)
    ax.set(xlabel="Nutrient", ylabel="Portion of goal consumed", title="User "+str(userID))
    plt.savefig("data/userAnalysis/user"+str(userID)+"/nutrientBoxplot.png"); plt.close()

    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(10,5))
    x = np.linspace(1, len(userGoals), len(userGoals))
    axes[0][0].plot(x, userGoals.goal_calories, label="calories", color='tab:blue')
    axes[0][0].plot(x, userGoals.goal_sodium, label="sodium", color='tab:orange')
    axes[1][0].plot(x, userGoals.goal_carbs, label="carbs", color='tab:green')
    axes[1][0].plot(x, userGoals.goal_fat, label="fat", color='tab:red')
    axes[1][0].plot(x, userGoals.goal_protein, label="protein", color='tab:purple')
    axes[1][0].plot(x, userGoals.goal_sugar, label="sugar", color='tab:brown')
    axes[0][0].set_ylabel("Daily nutrient goal"); axes[1][0].set_ylabel("Daily nutrient goal")
    axes[1][0].set_xlabel("Day")

    axes[0][1].plot(x, userGoals.calories, label="calories", color='tab:blue')
    axes[0][1].plot(x, userGoals.sodium, label="sodium", color='tab:orange')
    axes[1][1].plot(x, userGoals.carbs, label="carbs", color='tab:green')
    axes[1][1].plot(x, userGoals.fat, label="fat", color='tab:red')
    axes[1][1].plot(x, userGoals.protein, label="protein", color='tab:purple')
    axes[1][1].plot(x, userGoals.sugar, label="sugar", color='tab:brown')
    axes[0][1].set_ylabel("Nurtient consumption"); axes[1][1].set_ylabel("Nurtient consumption")
    axes[1][1].set_xlabel("Day")

    lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0][0], axes[1][0]]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', ncol=3)
    plt.savefig("data/userAnalysis/user"+str(userID)+"/nutrientTrends.png"); plt.close()

def new_fuct():
	return 0