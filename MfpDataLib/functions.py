import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB
from scipy.signal import find_peaks
from tqdm import tqdm

from pathlib import Path
basepath = Path(__file__).parent.parent

#### Data Processing ####

def process_myfitnesspal():
	(basepath/"data/myfitnesspal").mkdir(exist_ok=True, parents=True)
	(basepath/"data/myfitnesspal/raw").mkdir(exist_ok=True)

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
	df.to_csv(basepath/"data/raw/myfitnesspal_meals.csv", index=False)

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
	df.to_csv(basepath/"data/raw/myfitnesspal_goals.csv", index=False)

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

def count_stable_periods_and_points(data, threshold=10, min_length_percent=0.1):
    data = data.dropna()
    min_length = round(len(data) * min_length_percent)
    num_stable_periods = 0
    start = 0
    stable_indicies = []

    while start < len(data):
        end = start
        while end < len(data) - 1 and abs(data.iloc[end + 1] - data.iloc[start]) <= threshold:
            end += 1
        if end - start + 1 >= min_length:
            num_stable_periods += 1
            stable_indicies.extend(range(start, end + 1))
        start = end + 1

    return num_stable_periods,stable_indicies

def process_cohorts(goals_df, ID_list):

    # Create empty cohort lists
    manual_goal_no_change = []
    manual_goal_change = []
    automatic_goal_change = []
    automatic_goal_no_change = []

    for user_id in ID_list:

        # Filter data for the current person_id
        person_data = goals_df[goals_df['user_id'] == user_id]

        # Manual Goal with no change
        if np.std(person_data['goal_calories']) == 0: 
            manual_goal_no_change.append(user_id)
                
        # Manual Goal with Change
        num_stable_periods, stable_indicies = count_stable_periods_and_points(person_data['goal_calories'])
        stable_percent = len(stable_indicies)/len(person_data['goal_calories'])
            
        if user_id not in manual_goal_no_change:
            if num_stable_periods >= 2 and stable_percent > 0.5: 
                manual_goal_change.append(user_id)
            if num_stable_periods == 1 and stable_percent > .75:
                manual_goal_change.append(user_id)
            
        # Automatic Goal with Change 
        non_na_goal_calories = person_data['goal_calories'].dropna()
        minima_indices = find_peaks(-non_na_goal_calories)[0]
        local_minima_list = non_na_goal_calories.iloc[minima_indices].tolist()
        local_minima_list = [x + 1e-6 if np.abs(x) < 1e-3 else x for x in local_minima_list]
  
        percentage_change = np.diff(local_minima_list)/local_minima_list[:-1]*100

        exceed_threshold = np.abs(percentage_change) > 10 
            
        if any(exceed_threshold) and user_id not in manual_goal_change: 
            automatic_goal_change.append(user_id)
            
        # Calorie goal Patterns Automatic with change
        if user_id not in manual_goal_no_change and user_id not in automatic_goal_change and user_id not in manual_goal_change: 
            automatic_goal_no_change.append(user_id)

    file_path = 'data/cohorts/manual_goal_change_list.txt'
    with open(file_path, 'w') as file: 
        for person_id in manual_goal_change:
            file.write(f"{person_id}\n")
        print("Manual Goal w/ Change Cohort: ", str(len(manual_goal_change)), "(", str(np.round(100*len(manual_goal_change)/len(goals_df['user_id'].unique()), 2)), "%)")
                
    file_path = 'data/cohorts/manual_goal_no_change_list.txt'
    with open(file_path, 'w') as file: 
        for person_id in manual_goal_no_change:
            file.write(f"{person_id}\n")
        print("Manual Goal w/o Change Cohort: ", str(len(manual_goal_no_change)), "(", str(np.round(100*len(manual_goal_no_change)/len(goals_df['user_id'].unique()), 2)), "%)")
                
    file_path = 'data/cohorts/automatic_goal_no_change_list.txt'
    with open(file_path, 'w') as file: 
        for person_id in automatic_goal_no_change:
            file.write(f"{person_id}\n")
        print("Automatic Goal w/o Change Cohort: ", str(len(automatic_goal_no_change)), "(", str(np.round(100*len(automatic_goal_no_change)/len(goals_df['user_id'].unique()), 2)), "%)")
                
    file_path = 'data/cohorts/automatic_goal_change_list.txt'
    with open(file_path, 'w') as file: 
        for person_id in automatic_goal_change:
            file.write(f"{person_id}\n")
        print("Automatic Goal w Change Cohort: ", str(len(automatic_goal_change)), "(", str(np.round(100*len(automatic_goal_change)/len(goals_df['user_id'].unique()), 2)), "%)")

    return 

def standardize_nutrients(just_food, chunk_size=1000):
    # Create a database of nutrients normalized per 100 calories
    norm_food = pd.DataFrame({
        'full_name': just_food['full_name'],
        'calories': just_food['calories'],
        'carbs/100cal': 100 * just_food['carbs'] / just_food['calories'],
        'fat/100cal': 100 * just_food['fat'] / just_food['calories'],
        'protein/100cal': 100 * just_food['protein'] / just_food['calories'],
        'sodium/100cal': 100 * just_food['sodium'] / just_food['calories'],
        'sugar/100cal': 100 * just_food['sugar'] / just_food['calories']
    })

    # Create new dataframe with average nutrients across identical food names
    avg_nuts = norm_food.groupby('full_name').mean().reset_index()

    nutrients = ['carbs/100cal', 'fat/100cal', 'protein/100cal', 'sodium/100cal', 'sugar/100cal']
    
    print("Precomputing value counts...")
    value_counts = {}
    for col in tqdm(['serving_size', 'food_name', 'brand', 'flavor'], desc="Columns"):
        def safe_mode(x):
            counts = x.value_counts()
            return counts.index[0] if not counts.empty else np.nan
        
        value_counts[col] = just_food.groupby('full_name')[col].apply(safe_mode)
    
    print("Adding precomputed values to avg_nuts...")
    for col, counts in value_counts.items():
        avg_nuts[col] = avg_nuts['full_name'].map(counts)
    
    print("Initializing columns for best nutrients...")
    for nut in nutrients:
        avg_nuts[f'best{nut}'] = np.nan
    
    def process_group(group, name, avg_row):
        if group[nutrients].nunique().eq(1).all():
            food_ref = group.iloc[0][nutrients]
        else:
            dists = ((group[nutrients] - avg_row[nutrients]) / avg_row[nutrients]).abs().sum(axis=1)
            food_ref = group.loc[dists.idxmin(), nutrients]
        
        calories = just_food[just_food['full_name'] == name]['calories'].value_counts().index[0]
        return food_ref * (calories / 100)
    
    print("Processing groups in chunks...")
    full_names = list(norm_food.groupby('full_name').groups.keys())
    
    for i in tqdm(range(0, len(full_names), chunk_size), desc="Chunks"):
        chunk = full_names[i:i+chunk_size]
        chunk_groups = {name: norm_food.groupby('full_name').get_group(name) for name in chunk}
        
        for name, group in chunk_groups.items():
            avg_row = avg_nuts[avg_nuts['full_name'] == name].iloc[0]
            best_nuts = process_group(group, name, avg_row)
            
            for nut in nutrients:
                avg_nuts.loc[avg_nuts['full_name'] == name, f'best{nut}'] = best_nuts[nut]
        
        # Clear memory
        del chunk_groups
    
    print("Finalizing best_nuts dataframe...", end=" ")
    food_ref = avg_nuts.drop(columns=nutrients).dropna(how='any')
    # Rename columns
    food_ref = food_ref.rename(columns={
        'bestcarbs/100cal': 'carbs', 
        'bestfat/100cal': 'fat', 
        'bestprotein/100cal': 'protein', 
        'bestsodium/100cal': 'sodium',
        'bestsugar/100cal': 'sugar'
    })
    # Rearrange columns
    food_ref = food_ref[['full_name', 'food_name', 'brand', 'flavor', 'serving_size', 
                         'calories', 'carbs', 'fat', 'protein', 'sodium', 'sugar']]
    
    print('Done.')
    return food_ref

def compile_standard_nuts():
     # Read all csv files from reference folder
    files = [f for f in (basepath/"data/ref").iterdir() if f.is_file() and f.suffix == '.csv']
    files.sort()
    temp_ref_df = pd.DataFrame()
    for file in files:
        if file != '/home/emmett/Projects/Precision Nutrition/IO-with-behavior-Change/data/ref/standard_nutrient_reference.csv':
            print(file, end=": ")
            ref_batch = pd.read_csv(file)
            # Add to dataframe
            print(len(ref_batch))
            temp_ref_df = pd.concat([temp_ref_df, ref_batch], ignore_index=True)
    print(len(temp_ref_df), "total food item references")
    temp_ref_df = temp_ref_df.dropna(subset=['full_name']); temp_ref_df = temp_ref_df.drop_duplicates()
    print(len(temp_ref_df), "food item references")
    # Filter for just full_name with no duplicates
    food_names = temp_ref_df['full_name']
    print(str(len(temp_ref_df) - len(np.unique(food_names))), "duplicates to reprocess")
    ref_df = temp_ref_df.drop_duplicates(subset='full_name')
    print(len(ref_df))
    # Get the list of full_names which are not in the reference dataframe (so they were duplicates and need to be reprocessed)
    reprocess_full_names = list(set(temp_ref_df['full_name']) - set(ref_df['full_name']))
    print(reprocess_full_names)
    # Reprocess the full_names which were duplicates
    reprocess_df = temp_ref_df[temp_ref_df['full_name'].isin(reprocess_full_names)]
    reprocess_df = standardize_nutrients(reprocess_df)
    
    # Concatenate the reprocessed dataframe with the original reference dataframe
    ref_df = pd.concat([ref_df, reprocess_df], ignore_index=True)
    ref_df = ref_df.sort_values(by=['full_name'])
    ref_df.to_csv(basepath/"data/ref/standard_nutrient_reference.csv", index=False)

    return ref_df

def standardize_meals(meals_df, ref_df):
    
    return meals_df
     
#### Data Exploration ####

# Adjusted from Natalia's code:
def calc_user_stats(user_id, goals_df, meals_df, stats_df):

    user_goals = goals_df[goals_df['user_id'] == user_id].copy()

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
        print("User", str(userID), "has", len(userGoals), "daily entries with", len(userMeals["full_name"].unique()), "total unique food items")

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

#### Inverse Optimization and Inverse Learning Models ####

# Get the IO model parameters (A,b,X) from mfp dataframes
def model_params(goals_df, meals_df, dates, UB_flex, LB_flex):
    meals_df = meals_df[meals_df.date.isin(dates)] 
    foods = meals_df["full_name"].unique()
    print(len(meals_df), "food entries" , end=" (")

    # Observations
    X = np.empty(shape=(len(dates), len(foods)))
    f = 0
    for food in foods:
        d = 0
        for date in dates:
            servings = sum(meals_df[(meals_df.date.isin([date])) & (meals_df["full_name"]==food)]["num_servings"])
            X[d,f] = servings
            d += 1
        f += 1

    # TODO: Take this as a subset of a larger dictionary of references for food nutrients
    # A: Nutrients
    AT = meals_df.drop(columns=['user_id', 'date', 'food_name', 'brand', 'flavor', 'serving_size'])
    AT = AT.drop_duplicates(subset=["full_name"]); AT = AT.drop(columns='full_name')
    print(AT.shape[0], "unique foods) eaten in date range.")
    AT = AT[['calories', 'carbs', 'fat', 'protein', 'sodium', 'sugar']]
    AT = AT.astype('float').to_numpy()
    AT  = np.column_stack((AT, -AT))
    A = AT.T; A = np.nan_to_num(A)

    # b: Diet Bounds
    goals_df = goals_df[goals_df.date.isin(dates)]
    if len(goals_df) != 0:
        # +5% / -15% of calories goal
        cal_UBs = np.nan_to_num(np.reshape(goals_df['goal_calories'] + UB_flex[0]*goals_df['goal_calories'], newshape=(1,len(goals_df))), nan=9999)
        cal_LBs = np.nan_to_num(np.reshape(np.maximum(goals_df['goal_calories'] - LB_flex[0]*goals_df['goal_calories'], np.zeros(len(goals_df))), newshape=(1,len(goals_df))))
        # +5% / -25% of carbs goal
        carb_UBs = np.nan_to_num(np.reshape(goals_df['goal_carbs'] + UB_flex[1]*goals_df['goal_carbs'], newshape=(1,len(goals_df))), nan=999)
        carb_LBs = np.nan_to_num(np.reshape(np.maximum(goals_df['goal_carbs'] - LB_flex[1]*goals_df['goal_carbs'], np.zeros(len(goals_df))), newshape=(1,len(goals_df))))
        # +5% / -25% of fat goal
        fat_UBs = np.nan_to_num(np.reshape(goals_df['goal_fat'] + UB_flex[2]*goals_df['goal_fat'], newshape=(1,len(goals_df))), nan=999)
        fat_LBs = np.nan_to_num(np.reshape(np.maximum(goals_df['goal_fat'] - LB_flex[2]*goals_df['goal_fat'], np.zeros(len(goals_df))), newshape=(1,len(goals_df))))
        # +100% / -5% of protein goal
        prot_UBs = np.nan_to_num(np.reshape(goals_df['goal_protein'] + UB_flex[3]*goals_df['goal_protein'], newshape=(1,len(goals_df))), nan=999)
        prot_LBs = np.nan_to_num(np.reshape(np.maximum(goals_df['goal_protein'] - LB_flex[3]*goals_df['goal_protein'], np.zeros(len(goals_df))), newshape=(1,len(goals_df))))
        # +5% / -50% of sodium goal
        sod_UBs = np.nan_to_num(np.reshape(goals_df['goal_sodium'] + UB_flex[4]*goals_df['goal_sodium'], newshape=(1,len(goals_df))), nan=9999)
        sod_LBs = np.nan_to_num(np.reshape(np.maximum(goals_df['goal_sodium'] - LB_flex[4]*goals_df['goal_sodium'], np.zeros(len(goals_df))), newshape=(1,len(goals_df))))
        # +5% / 0 min of sugar goal
        sug_UBs = np.nan_to_num(np.reshape(goals_df['goal_sugar'] + UB_flex[5]*goals_df['goal_sugar'], newshape=(1,len(goals_df))), nan=999)
        sug_LBs = np.reshape(np.zeros(len(goals_df)), newshape=(1,len(goals_df)))

        # TEMP: remove sugar bounds for debugging
        b = np.concat([cal_UBs, carb_UBs, fat_UBs, prot_UBs, sod_UBs, sug_UBs,
					   -cal_LBs, -carb_LBs, -fat_LBs, -prot_LBs, -sod_LBs, -sug_LBs], axis=0).T

    else:
        b = None
        print("No data for day(s)")

    return A, b, X, foods

# Modular multi-space Inverse Optimization model
def IO_M(A,b,x,myEnv,noiseType, diffType="None", diff =-999999, e_x_abs_max=-999999, e_A_abs_max=-999999,
         x_noise=False, A_noise=False, b_noise=False, c_u=False, y_u=False, x_bar_u=False, A_bar_u=False, b_bar_u=False, corner_u=False):
    
    # Problem dimensions
    m = A.shape[1] 
    n = A.shape[2] 
    p = A.shape[0]
    if p != x.shape[0]: print("Error: Dimensions of X and As must match"); return 0

    # Model settings
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        mod = gp.Model(env=myEnv)
        mod.Params.OutputFlag = 0
        mod.Params.timeLimit = 100.0
        mod.Params.DualReductions = 0 
        mod.params.NonConvex = 2  
        
        # Create variables
        c = mod.addVars(1,n, lb=-GRB.INFINITY) if c_u else mod.addVars(p,n, lb=-GRB.INFINITY)
        y = mod.addVars(1,m, lb=0) if y_u else mod.addVars(p,m, lb=0)
        x_bar = mod.addVars(1,n, lb=-GRB.INFINITY) if x_bar_u else mod.addVars(p,n, lb=-GRB.INFINITY)
        A_bar = mod.addVars(1,m,n, lb=-GRB.INFINITY) if A_bar_u else mod.addVars(p,m,n, lb=-GRB.INFINITY)
        b_bar = mod.addVars(1,m, lb=-GRB.INFINITY) if b_bar_u else mod.addVars(p,m, lb=-GRB.INFINITY)
        # Additional variables
        c_abs = mod.addVars(p,n, lb=0)
        e_x = mod.addVars(p,n, lb=-GRB.INFINITY); e_x_abs = mod.addVars(p,n, lb=0) 
        e_A = mod.addVars(p,m,n, lb=-GRB.INFINITY); e_A_abs = mod.addVars(p,m,n, lb=0) 
        e_b = mod.addVars(p,m, lb=-GRB.INFINITY); e_b_abs = mod.addVars(p,m, lb=0)
        v = mod.addVars(1, m, vtype=GRB.BINARY, name='v')

        # Set objective:
        if noiseType == 'scl':
            e_x_loss = mod.addVars(p,n, lb=-GRB.INFINITY); e_A_loss = mod.addVars(p,m,n, lb=-GRB.INFINITY); e_b_loss = mod.addVars(p,m, lb=-GRB.INFINITY)
            mod.setObjective(gp.quicksum(e_A_loss[k,i,j] for k in range(p) for i in range(m) for j in range(n)) +
                             gp.quicksum(e_x_loss[k,j] for k in range(p) for j in range(n)) + 
                             gp.quicksum(e_b_loss[k,i] for k in range(p) for i in range(m)), GRB.MINIMIZE)
        else:
            mod.setObjective(gp.quicksum(e_A_abs[k,i,j] for k in range(p) for i in range(m) for j in range(n)) +
                             gp.quicksum(e_x_abs[k,j] for k in range(p) for j in range(n)) + 
                             gp.quicksum(e_b_abs[k,i] for k in range(p) for i in range(m)), GRB.MINIMIZE)

        # Constraints
        for k in range (p):
            #### Unified parameter conditions ####
            k_c = 0 if c_u else k; k_y = 0 if y_u else k; k_x = 0 if x_bar_u else k; k_A = 0 if A_bar_u else k; k_b = 0 if b_bar_u else k
            
            #### Noised Parameters ####
            if noiseType == "scl":
                if not x_noise: mod.addConstrs(e_x[k,j]==1 for j in range(n))
                if not A_noise: mod.addConstrs(e_A[k,i,j]==1 for i in range(m) for j in range(n))
                if not b_noise: mod.addConstrs(e_b[k,i]==1 for i in range(m))
                
                # Decision-space error
                mod.addConstrs(x[k,j] * e_x[k,j] == x_bar[k_x,j] for j in range(n))
                mod.addConstrs(e_x_loss[k,j] >= e_x[k,j] - 1 for j in range(n))
                mod.addConstrs(e_x_loss[k,j] >= 1 - e_x[k,j] for j in range(n))
                # Constraint-space error (A / LHS)
                mod.addConstrs(A[k,i,j] * e_A[k,i,j] == A_bar[k_A,i,j] for i in range(m) for j in range(n))
                mod.addConstrs(e_A_loss[k,i,j] >= e_A[k,i,j] - 1 for i in range(m) for j in range(n))
                mod.addConstrs(e_A_loss[k,i,j] >= 1 - e_A[k,i,j] for i in range(m) for j in range(n))
                # Constraint-space error (b / RHS) 
                mod.addConstr(b[k,i] * e_b[k,i] == b_bar[k_b,i])
                mod.addConstr(e_b_loss[k,i] >= e_b[k,i] - 1)
                mod.addConstr(e_b_loss[k,i] >= 1 - e_b[k,i])
            else:
                if not x_noise: mod.addConstrs(e_x[k,j]==0 for j in range(n))
                if not A_noise: mod.addConstrs(e_A[k,i,j]==0 for i in range(m) for j in range(n))
                if not b_noise: mod.addConstrs(e_b[k,i]==0 for i in range(m))
                if noiseType == "abs":
                    # Decision-space error
                    mod.addConstrs(x[k,j] + e_x[k,j] == x_bar[k_x,j] for j in range(n))
                    # Constraint-space error (A / LHS)
                    # TODO: Fix "GurobiError: Constant is Nan"
                    mod.addConstrs(A[k,i,j] + e_A[k,i,j] == A_bar[k_A,i,j] for i in range(m) for j in range(n))
                    # Constraint-space error (b / RHS)
                    mod.addConstrs(b[k,i] + e_b[k,i] == b_bar[k_b,i] for i in range(m))
                elif noiseType == "rel":
                    # Decision-space error
                    mod.addConstrs(x[k,j] + x[k,j] * e_x[k,j] == x_bar[k_x,j] for j in range(n))
                    # Constraint-space error (A / LHS)
                    mod.addConstrs(A[k,i,j] + A[k,i,j] * e_A[k,i,j] == A_bar[k_A,i,j] for i in range(m) for j in range(n))
                    # Constraint-space error (b / RHS)
                    mod.addConstrs(b[k,i] + b[k,i] * e_b[k,i] == b_bar[k_b,i] for i in range(m))
                else: print("Error: Please enter valid noiseType (abs, rel, scl)"); return 
            # Absolute values
            mod.addConstrs(e_x_abs[k,j] == gp.abs_(e_x[k,j]) for j in range(n))
            mod.addConstrs(e_A_abs[k,i,j] == gp.abs_(e_A[k,i,j]) for i in range(m) for j in range(n))
            mod.addConstrs(e_b_abs[k,i] == gp.abs_(e_b[k,i]) for i in range(m))

            #### Noise-type balance/difference, if enforced ####
            if diffType=='abs_diff':
                # Set difference between absolute constraint-space and absolute decision-space error
                if diff==-999999: print("Error: Need to set diff for abs_diff"); return
                if e_x_abs_max==-999999 or e_A_abs_max==-999999: print("Error: Need to set e_x_abs_max and e_A_abs_max for abs_diff"); return
                if noiseType=="scl":
                    mod.addConstr(gp.quicksum(e_A_loss[k,i,j] for k in range(p) for i in range(m) for j in range(n)) - gp.quicksum(e_x_loss[k,j] for k in range(p) for j in range(n)) == diff)
                    mod.addConstr(gp.quicksum(e_x_loss[k,j] for k in range(p) for j in range(n)) <= e_x_abs_max)
                    mod.addConstr(gp.quicksum(e_A_loss[k,i,j] for k in range(p) for i in range(m) for j in range(n)) <= e_A_abs_max)
                else:
                    mod.addConstr(gp.quicksum(e_A_abs[k,i,j] for k in range(p) for i in range(m) for j in range(n)) - gp.quicksum(e_x_abs[k,j] for k in range(p) for j in range(n)) == diff)
                    mod.addConstr(gp.quicksum(e_x_abs[k,j] for k in range(p) for j in range(n)) <= e_x_abs_max)
                    mod.addConstr(gp.quicksum(e_A_abs[k,i,j] for k in range(p) for i in range(m) for j in range(n)) <= e_A_abs_max)
            elif diffType =='rel_diff':
                # Set relative absolute constraint-space and absolute decision-space error
                if diff==-999999: print("Error: Need to set diff for rel_diff"); return
                if noiseType=="scl":
                    mod.addConstr(gp.quicksum(e_A_loss[k,i,j] for k in range(p) for i in range(m) for j in range(n)) == 
                        diff*(gp.quicksum(e_A_loss[k,i,j] for k in range(p) for i in range(m) for j in range(n))
                        +gp.quicksum(e_x_loss[k,j] for k in range(p) for j in range(n))))
                else:
                    mod.addConstr(gp.quicksum(e_A_abs[k,i,j] for k in range(p) for i in range(m) for j in range(n)) == 
                        diff*(gp.quicksum(e_A_abs[k,i,j] for k in range(p) for i in range(m) for j in range(n))
                        +gp.quicksum(e_x_abs[k,j] for k in range(p) for j in range(n))))
            
            #### General IO Constraints ####
            # Strong duality
            mod.addConstr(gp.quicksum(c[k_c,j] * x_bar[k_x,j] for j in range(n)) == gp.quicksum(b_bar[k_b,i] * y[k_y,i] for i in range(m)))
            # Primal feasibility
            mod.addConstrs(gp.quicksum(A_bar[k_A,i,j] * x_bar[k_x,j] for j in range(n)) <= b_bar[k_b,i] for i in range(m))
            mod.addConstrs(x_bar[k_x,j] >= 0 for j in range(n))
            # Dual feasibility
            mod.addConstrs(gp.quicksum(y[k_y,i] * A_bar[k_A,i,j] for i in range(m)) == c[k_c,j] for j in range(n))    
            # 1-Norm of c
            mod.addConstr(gp.quicksum(c_abs[k_c,j] for j in range(n)) == 1) 
            mod.addConstrs(c_abs[k_c,j] == gp.abs_(c[k_c,j]) for j in range(n))
            # Extreme point solution
            if corner_u:
                mod.addConstrs(gp.quicksum(A_bar[k_A,i,j] * x_bar[k_x,j] for j in range(n)) <= b_bar[k_b,i] + 999999999*(1-v[0,i]) for i in range(m))   
                mod.addConstr(gp.quicksum(v[0,i] for i in range(m)) == n)
        
        # Optimize 
        mod.optimize()
        if mod.status == GRB.OPTIMAL:
            print('Optimal objective value = ', mod.ObjVal)
            c_vals = np.zeros(shape=(p,n))
            y_vals = np.zeros(shape=(p,m))
            x_bar_vals = np.zeros(shape=(p,n)); e_x_vals = np.zeros(shape=(p,n))
            A_bar_vals = np.zeros(shape=(p,m,n)); e_A_vals = np.zeros(shape=(p,m,n))
            b_bar_vals = np.zeros(shape=(p,m)); e_b_vals = np.zeros(shape=(p,m))
            for k in range(p):
                # Unified parameter conditions
                k_c = 0 if c_u else k; k_y = 0 if y_u else k; k_x = 0 if x_bar_u else k; k_A = 0 if A_bar_u else k; k_b = 0 if b_bar_u else k
                for i in range(m):
                    y_vals[k,i] = y[k_y,i].X
                    b_bar_vals[k,i] = b_bar[k_b,i].X; e_b_vals[k,i] = e_b[k,i].X
                    for j in range(n):
                        A_bar_vals[k,i,j] = A_bar[k_A,i,j].X; e_A_vals[k,i,j] = e_A[k,i,j].X
                for j in range(n):
                    c_vals[k,j] = c[k_c,j].X
                    x_bar_vals[k,j] = x_bar[k_x,j].X; e_x_vals[k,j] = e_x[k,j].X
            return {'feasible': True, 'obj': mod.ObjVal, 'c': c_vals, 'y': y_vals, 'x_bar': x_bar_vals, 'e_x': e_x_vals,
                    'A_bar': A_bar_vals, 'e_A': e_A_vals, 'b_bar': b_bar_vals, 'e_b': e_b_vals, 'rt': mod.Runtime}
        else: print("Optimization ended with status ", mod.status); return {'feasible': False} 

# Plot nutritional changes in recommendation vs observation
def plt_rec_nuts(userID, userGoals, dates, A, b, X, d, day_ind, all_results, model_names):
    # Create the bar plot and set up formatting
    plt.figure(figsize=(12, 6)); plt.grid(visible=True, axis='y'); plt.rcParams['font.size'] = 10
    num_bars = len(all_results) + 1
    bar_width = 1/(num_bars+1)
    # Set the positions of the bars on the x-axis
    nutrients = ["Calories", "Carbs", "Fat", "Protein", "Sodium", "Sugar"]
    start_pos = range(len(nutrients)) - 0.5*np.ones(len(nutrients)) + bar_width*np.ones(len(nutrients))
    positions = [start_pos]
    for i in range(len(all_results)):   
        positions.append([x + (i+1)*bar_width for x in start_pos])

    # Get actual original behavior amounts
    obs_cals = gp.quicksum(X[d,i]*A[0,i] for i in range(X.shape[1])).getValue()
    goal_cals = userGoals.iloc[day_ind]["goal_calories"]
    obs_carbs = gp.quicksum(X[d,i]*A[1,i] for i in range(X.shape[1])).getValue()
    goal_carbs = userGoals.iloc[day_ind]["goal_carbs"]
    obs_fat = gp.quicksum(X[d,i]*A[2,i] for i in range(X.shape[1])).getValue()
    goal_fat = userGoals.iloc[day_ind]["goal_fat"]
    obs_protein = gp.quicksum(X[d,i]*A[3,i] for i in range(X.shape[1])).getValue()
    goal_protein = userGoals.iloc[day_ind]["goal_protein"]
    obs_sodium = gp.quicksum(X[d,i]*A[4,i] for i in range(X.shape[1])).getValue()
    goal_sodium = userGoals.iloc[day_ind]["goal_sodium"]
    obs_sugar = gp.quicksum(X[d,i]*A[5,i] for i in range(X.shape[1])).getValue()
    goal_sugar = userGoals.iloc[day_ind]["goal_sugar"]
    obs_df = pd.DataFrame({"Day": d, "Calories": obs_cals, "Carbs": obs_carbs, "Fat": obs_fat, "Protein": obs_protein, "Sodium": obs_sodium, "Sugar": obs_sugar}, index=[0])
    goals_df = pd.DataFrame({"Day": d, "Calories": goal_cals, "Carbs": goal_carbs, "Fat": goal_fat, "Protein": goal_protein, "Sodium": goal_sodium, "Sugar": goal_sugar}, index=[0])
    # Plot bars for original behavior
    container = plt.bar(positions[0], 100*(obs_df.iloc[0,1:] - goals_df.iloc[0,1:])/goals_df.iloc[0,1:], width=bar_width, label='Original Behavior')
    plt.bar_label(container, labels=obs_df.iloc[0,1:].astype(int))

    # Get recommended amounts for each model:
    for m in range(len(all_results)):
        rec_cals = gp.quicksum(all_results[m]['x_bar'][d,j]*(all_results[m]['A_bar'][0,0,j]) for j in range(X.shape[1])).getValue()
        rec_carbs = gp.quicksum(all_results[m]['x_bar'][d,j]*(all_results[m]['A_bar'][0,1,j]) for j in range(X.shape[1])).getValue()
        rec_fat = gp.quicksum(all_results[m]['x_bar'][d,j]*(all_results[m]['A_bar'][0,2,j]) for j in range(X.shape[1])).getValue()
        rec_protein = gp.quicksum(all_results[m]['x_bar'][d,j]*(all_results[m]['A_bar'][0,3,j]) for j in range(X.shape[1])).getValue()
        rec_sodium = gp.quicksum(all_results[m]['x_bar'][d,j]*(all_results[m]['A_bar'][0,4,j]) for j in range(X.shape[1])).getValue()
        rec_sugar = gp.quicksum(all_results[m]['x_bar'][d,j]*(all_results[m]['A_bar'][0,5,j]) for j in range(X.shape[1])).getValue()
        recs_df = pd.DataFrame({"Day": d, "Calories": rec_cals, "Carbs": rec_carbs, "Fat": rec_fat, "Protein": rec_protein, "Sodium": rec_sodium, "Sugar": rec_sugar}, index=[0])
        # Plot recommendation amounts for each model
        container = plt.bar(positions[m+1], 100*(recs_df.iloc[0,1:] - goals_df.iloc[0,1:])/goals_df.iloc[0,1:], width=bar_width, label=model_names[m])
        plt.bar_label(container, labels=recs_df.iloc[0,1:].astype(int))

    # Define the acceptable range for each nutrient
    day_goals = userGoals[userGoals.date.isin(dates)].iloc[0]
    acceptable_ranges = {
        'Calories': (100*(-b[6] - day_goals['goal_calories'])/day_goals['goal_calories'], 
                     100*(b[0] - day_goals['goal_calories'])/day_goals['goal_calories']),
        'Carbs': (100*(-b[7] - day_goals['goal_carbs'])/day_goals['goal_carbs'], 
                  100*(b[1] - day_goals['goal_carbs'])/day_goals['goal_carbs']),
        'Fat': (100*(-b[8] - day_goals['goal_fat'])/day_goals['goal_fat'], 
                100*(b[2] - day_goals['goal_fat'])/day_goals['goal_fat']),
        'Protein': (100*(-b[9] - day_goals['goal_protein'])/day_goals['goal_protein'], 
                    100*(b[3] - day_goals['goal_protein'])/day_goals['goal_protein']),
        'Sodium': (100*(-b[10] - day_goals['goal_sodium'])/day_goals['goal_sodium'], 
                   100*(b[4] - day_goals['goal_sodium'])/day_goals['goal_sodium']),
        'Sugar': (100*(-b[11] - day_goals['goal_sugar'])/day_goals['goal_sugar'], 
                  100*(b[5] - day_goals['goal_sugar'])/day_goals['goal_sugar'])}

    # Add green bands for acceptable ranges
    nutrient_goals = []
    for i, nutrient in enumerate(nutrients):
        if np.isnan(goals_df.iloc[0][nutrient]): nutrient_goals.append("None")
        else: nutrient_goals.append(str(round(goals_df.iloc[0][nutrient])))
        lower_bound, upper_bound = acceptable_ranges[nutrient]
        plt.axvline(x=i-0.5, color='k', linestyle='--', linewidth=0.5)
        plt.axvline(x=i+0.5, color='k', linestyle='--', linewidth=0.5)
        plt.axhspan(lower_bound, upper_bound, facecolor='g', alpha=0.2, xmin=i/6, xmax=(i+1)/6)

    # Add labels and title
    plt.xlabel('Nutrients')
    plt.ylabel('Difference from goal (%)')
    plt.title('Day '+ str(d+1) +' Nutrient Recommendations Results')
    nutrient_labels = ["Calories ("+nutrient_goals[0]+")", "Carbs ("+nutrient_goals[1]+")", "Fat ("+nutrient_goals[2]+")", 
                       "Protein ("+nutrient_goals[3]+")", "Sodium ("+nutrient_goals[4]+")", "Sugar ("+nutrient_goals[5]+")"]
    plt.xticks([r  for r in range(len(nutrients))], nutrient_labels); # plt.tick_params(axis='x', labelsize=12)
    plt.xlim(-0.5, len(nutrients)-0.5)
    plt.legend(bbox_to_anchor=(0.75, 1.15), loc="upper left")
    plt.savefig("data/User"+str(userID)+"_Day_"+str(d+1)+"_Nutrients.png")
    
    # plt.close()
    return

# Plot food portion changes in recommendation vs observation
def plt_rec_foods(userID, X, foods, d, all_results, model_names):
    # Get observed behavior
    obs = X[d].T!=0
    dayResults = {"Observation": X[d][obs]}
    rec_foods = foods[obs]
    # Get all foods recommended across all models (and the original observation)
    for i in range(len(all_results)):
        rec = all_results[i]['x_bar'][d].T!=0
        rec_foods = np.append(rec_foods, foods[rec])
    # Get boolean array for foods in rec_foods
    rec_foods = np.unique(np.array(rec_foods))
    food_bool = np.isin(foods, rec_foods)

    dayResults = {"Observation": X[d][food_bool]}
    for i in range(len(all_results)):
         dayResults[model_names[i]] = all_results[i]['x_bar'][d][food_bool]
         
    model_names.insert(0, "Observation")
    
    # Create the bar plot and set up formatting
    plt.figure(figsize=(12, 12)); plt.grid(visible=True, axis='y'); plt.rcParams['font.size'] = 6
    num_bars = len(all_results) + 1
    bar_width = 1/(num_bars+1)
    start_pos = range(len(rec_foods)) - 0.5*np.ones(len(rec_foods)) + bar_width*np.ones(len(rec_foods))
    positions = [start_pos]
    for i in range(len(all_results)):   
        positions.append([x + (i+1)*bar_width for x in start_pos])
    
    # Create the bar plot
    for i in range(len(dayResults)):
        plt.bar(positions[i], dayResults[model_names[i]], width=bar_width, label=model_names[i])

    # Add labels and title
    plt.xlabel('Food Groups')
    plt.ylabel('Values')
    plt.title('Day ' + str(d+1) + ' Food Groups vs. Values')
    plt.xticks([r for r in range(len(rec_foods))], rec_foods, rotation=90)
    plt.legend(bbox_to_anchor=(0.75, 1.2), loc="upper left")
    # Adjust layout 
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.6, top=0.925)
    # plt.tight_layout()
    
    plt.savefig("data/User"+str(userID)+"_Day_"+str(d+1)+"_Foods.png")
    # plt.close()
    return 

# Plot the stepwise adjustement recommendations from IL model(s)
def plt_stepwise(filepath):
    # Data from the dictionary
    data = pd.read_csv(filepath)
    food_items = list(data.columns)
    food_items_cleaned = [item.split(',')[0] for item in food_items]

    # Convert the data into a format suitable for a heatmap
    heatmap_data = np.array([data.iloc[i] for i in range(len(data))])

    # Create a heatmap
    plt.figure(figsize=(22, 8))
    plt.rcParams.update({'font.size': 14})
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', xticklabels=food_items_cleaned, 
                yticklabels=['Original Diet', 'First Adjustement', 'Second Adjustement', 
                            'Third Adjustement', 'Fourth Adjustement', 'Fifth Adjustement'])

    # Add labels and title
    plt.title('Low-Carb Day 1: Servings Recommended via Stepwise Adjustment')
    plt.xlabel('Food Items')

    # Show the plot
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()
    return 
