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

from pathlib import Path
basepath = Path(__file__).parent.parent

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
        print("User", str(userID), "has", len(userGoals), "daily entries with", len(userMeals["description"].unique()), "total unique food items")

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
    AT = AT[['calories', 'carbs', 'fat', 'protein', 'sodium']]
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

#### Inverse Optimization and Inverse Learning Models ####

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
            # print('Optimal objective value = ', mod.ObjVal)
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
        else: print(mod.status); return {'feasible': False} 