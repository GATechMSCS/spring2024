# load and preprocess
import pandas as pd

# model preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# model evaluation
from sklearn.metrics import accuracy_score


def clean(dataset:str) -> pd.DataFrame:

    match dataset:
        
        case "cvd":

            file1 = "cardio_data_processed.csv"
            cardio_vasc = pd.read_csv(file1)
            
            # id: just an index
            # useing age_years over age
            # using weight and height instead of bmi
            # using ap_hi/lo instead of bp_category and bp_category_encoded
            cardio_cols_drop = ['id', 'age', 'bmi', 'bp_category',
                                'bp_category_encoded', "alco", "smoke",
                                'gender', 'gluc', 'active', 'cholesterol']
            cardio_vasc = cardio_vasc.drop(columns=cardio_cols_drop, axis=1)

            cleaned = cardio_vasc.sample(frac=0.15)

        case "nf":

            file2 = "MyFoodData_Nutrition_Facts_SpreadSheet_Release_1.4.xlsx"
            nutrition_facts = pd.read_excel(file2)
    
            nutrition_facts = nutrition_facts.dropna(axis=0, subset=["Food Group"])

            # dropping columns that have every value missing
            cols_drop = ["Added Sugar g", "Soluble Fiber g", "Insoluble Fiber g",
                        "Total sugar alcohols g", "Molybdenum mcg", "Chlorine mg",
                        "Biotin B7 mcg", "NetCarbs g"]
            nutrition_facts = nutrition_facts.drop(columns=cols_drop, axis=1)

            # dropping cols that don't seem to mean much
            more_drop = ["PRAL score", "ID", "Name", '183 n3 ccc ALA mg',
                        '205 n3 EPA mg', '225 n3 DPA mg', '226 n3 DHA mg',
                        "Serving Weight 1 g", "Serving Weight 2 g", "Serving Weight 3 g",
                        "Serving Weight 4 g", "Serving Weight 5 g", "Serving Weight 6 g",
                        "Serving Weight 7 g", "Serving Weight 8 g", "Serving Weight 9 g",
                        "200 Calorie Weight g", "Saturated Fats g",
                        "Fat g", "Fiber g", "Calcium mg", "Iron Fe mg", "Potassium K mg", "Magnesium mg",
                        "Vitamin A RAE mcg", "Vitamin C mg", "Vitamin B12 mcg", "Vitamin D mcg",
                        "Vitamin E AlphaTocopherol mg", "Omega 3s mg", "Omega 6s mg", "Phosphorus P mg",
                        "Copper Cu mg", "Thiamin B1 mg", "Riboflavin B2 mg", "Vitamin B6 mg", "Folate B9 mcg",
                        "Folic acid mcg", "Food Folate mcg", "Folate DFE mcg", "Choline mg", "Retinol mcg",
                        "Carotene beta mcg", "Carotene alpha mcg", "Lycopene mcg", "Lutein + Zeaxanthin mcg",
                        "Vitamin K mcg", "Fatty acids total monounsaturated mg", "Fatty acids total polyunsaturated mg",
                        "Alcohol g", "Caffeine mg", "Theobromine mg", "Sugars g", "Niacin B3 mg",
                        "Selenium Se mcg", "Zinc Zn mg", "Calories"]
            nutrition_facts = nutrition_facts.drop(columns=more_drop, axis=1)

            # drop column if 70% of its rows are empty
            threshold = int(.70*len(nutrition_facts))
            nutrition_facts.dropna(axis=1, thresh=threshold, inplace=True)

            nutrition_facts.fillna(0, inplace=True)

            nutrition_facts.columns = nutrition_facts.columns.str.lower()

            cols_rename = {"food group": "food_group", "protein g": "protein", "carbohydrate g": "carbohydrate",
                            "cholesterol mg": "cholesterol", "water g": "water", "sodium mg": "sodium"}

            nutrition_facts = nutrition_facts.rename(mapper=cols_rename, axis=1)

            bool_mask1 = (nutrition_facts['food_group'] == 'Meats') | (nutrition_facts['food_group'] == 'Vegetables') 
            bool_mask2 = bool_mask1 | (nutrition_facts['food_group'] == 'Baked Foods')
            bool_mask3 = bool_mask2 | (nutrition_facts['food_group'] == 'Fish')
            cleaned = nutrition_facts[bool_mask3]
            
    return cleaned

def split(cleaned:pd.DataFrame,
          stratify:pd.Series,
          cols_drop:str,
          target_col:list,
          test_size=0.15,):

    # split data
    train, test = train_test_split(cleaned,
                                   test_size=test_size,
                                   random_state=123,
                                   stratify=stratify)

    # get X, y
    X_train = train.drop(cols_drop, axis=1)
    X_test = test.drop(cols_drop, axis=1)
    y_train = train[target_col]
    y_test = test[target_col]
    

    return X_train, X_test, y_train, y_test

def scale(X_train:pd.DataFrame, 
          X_test:pd.Series):
    
    scale = StandardScaler()
    scale.fit(X_train)
    X_train_scaled = pd.DataFrame(data=scale.transform(X_train),
                                    columns=X_train.columns,
                                    index=X_train.index)
    X_test_scaled = pd.DataFrame(data=scale.transform(X_test),
                                    columns=X_test.columns,
                                    index=X_test.index)

    
    return X_train_scaled, X_test_scaled

def baseline(y_train:pd.Series) -> float:

    # calculat a baseline
    act_pred_error = pd.DataFrame(data={"actual": y_train})
    act_pred_error["baseline_prediction"] = y_train.value_counts().index[0]

    baseline_acc = accuracy_score(act_pred_error["actual"], act_pred_error["baseline_prediction"])

    # print baseline accuracy
    print(f"\nBaseline Accuracy Score: {round(baseline_acc, 2)}%\n")

    return True
    
def final_dataset(dataset:str) -> pd.DataFrame:

    match dataset:
        
        case 'cvd':
            print(f'{dataset.upper()} Loading and Cleaning...')
            cleaned = clean(dataset=dataset)
            print(f'{dataset.upper()} Loaded and Cleaned...')

            print(f'\n{dataset.upper()} Splitting...')
            X_train, X_test, y_train, y_test = split(cleaned=cleaned,
                                                     stratify=cleaned['cardio'],
                                                     cols_drop=['cardio'],
                                                     target_col='cardio')
            print(f'{dataset.upper()} Split...')

            print(f'\n{dataset.upper()} Scaling...')
            X_train_scaled, X_test_scaled = scale(X_train, X_test)
            print(f'{dataset.upper()} Scaled...\n')

        case 'nf':
            print(f'{dataset.upper()} Loading and Cleaning...')
            cleaned = clean(dataset=dataset)
            print(f'{dataset.upper()} Loaded and Cleaned...')

            print(f'\n{dataset.upper()} Splitting...')
            X_train, X_test, y_train, y_test = split(cleaned=cleaned,
                                                     stratify=cleaned['food_group'],
                                                     cols_drop=['food_group'],
                                                     target_col='food_group')
            print(f'{dataset.upper()} Split...')

            print(f'\n{dataset.upper()} Scaling...')
            X_train_scaled, X_test_scaled = scale(X_train, X_test)
            print(f'{dataset.upper()} Scaled...')

    baseline(y_train)
        
    return X_train_scaled, X_test_scaled, y_train, y_test

def main():

    cvd = final_dataset(dataset='cvd')

    nf = final_dataset(dataset='nf')
    print('TEST')
    print(cvd)
    print()
    print(nf)
    print('TEST')

if __name__ == "__main__":
        main()