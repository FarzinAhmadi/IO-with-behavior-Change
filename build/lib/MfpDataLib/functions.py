import json
import pandas as pd

from pathlib import Path
basepath = Path(__file__).parent.parent

def say_hello(name):
	print("Hello, " + name + "! Nice to meet you :)")
	return 

def ask_hru():
	print("How are you???????")

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