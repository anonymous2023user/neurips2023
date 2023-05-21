import numpy as np
import pandas as pd
import itertools
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from tqdm import tqdm
from sklearn.utils import shuffle


# patients_list_ = np.array([1, 3, 5, 7, 8, 9, 10, 11, 13, 15, 18, 19, 22, 23]) - 1  # 14 patients with better results
patients_list_ = np.array([np.arange(0, 23)], dtype=object)  # updated file
# patients_list_ = np.array([np.arange(0, 14)], dtype=object)  # For MIT-BIH

patients_list = list(itertools.chain(*patients_list_))  # flattened
# patients_list = list(patients_list_)  # 14 patients with better results

patients_array = np.array(patients_list)
print(patients_array)

delta_list = [0.0001,	0.0005,	0.001,	0.0015,	0.002,	0.0025,	0.003,	0.0035,	0.004,	0.0045,	0.005]  # For CHB_MIT
# delta_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3,
#               0.32, 0.35, 0.38, 0.4, 0.42, 0.45, 0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]  # For MIT-BIH

# Without access to the training set
df = pd.read_excel('Data/test set/MMAttack_CNN (updated).xlsx', sheet_name=patients_list)   # updated file
# df = pd.read_excel('Data/test set/BIH_all.xlsx', sheet_name=patients_list)  # For MIT-BIH
# df = pd.read_excel('Data/test set/BIH_healthy.xlsx', sheet_name=patients_list)  # For MIT-BIH
# df = pd.read_excel('Data/test set/healthy_know_labels.xlsx', sheet_name=patients_list)  # For MIT-CHB 23p/14p (non-seizure sig.)

# With access to the training set
# df = pd.read_excel('Data/train set/CHB_all.xlsx', sheet_name=patients_list)   # updated file
# df = pd.read_excel('Data/train set/CHB_healthy.xlsx', sheet_name=patients_list)   # For MIT-CHB 23p/14p (non-seizure sig.)
# df = pd.read_excel('Data/train set/BIH_all.xlsx', sheet_name=patients_list)  # For MIT-BIH
# df = pd.read_excel('Data/train set/BIH_healthy.xlsx', sheet_name=patients_list)  # For MIT-BIH


def generate_data(patients_array, shuffle_flag=False):
    x = []
    y = []

    num_samples = 100  # number of samples for each patient

    for model in patients_array:
        patients = np.array(patients_list_)  # patients_array
        patients = list(np.delete(patients,  np.where(patients == model)[1][0]))  # For MIT-BIH (also 23pMMAttack_CNN)
        # patients = list(np.delete(patients, np.where(patients == model)))  # For 14 patients with better results
        for i in range(num_samples):
            random_patients = random.sample(patients, 4)  # the number of patients can be changed (1 to 22)
            random_patients.append(model)
            random.shuffle(random_patients)
            y.append(random_patients.index(model))


            sample = []
            for patient in random_patients:
                for delta in delta_list:
                    sample.append(df[model][delta].loc[patient]/df[model]['Total'].loc[patient])  # For CHB_MIT
                    # sample.append(df[model][delta].loc[patient] / df[model]['all'].loc[patient])  # For MIT-BIH

            x.append(np.array(sample))

    if shuffle_flag:
        x, y = shuffle(np.array(x), np.array(y))
    else:
        x, y = np.array(x), np.array(y)
    return x, y


def train_and_evaluate_ml_models(x_train, y_train, x_test, y_test, num_test_patients, num_samples):

    # train
    clf_ert = ExtraTreesClassifier(n_estimators=100).fit(x_train, y_train)
    clf_rf = RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
    clf_xgb = GradientBoostingClassifier(n_estimators=100).fit(x_train, y_train)

    # predict
    y_predicted_ert = clf_ert.predict(x_test)
    y_predicted_rf = clf_rf.predict(x_test)
    y_predicted_xgb = clf_xgb.predict(x_test)

    # see if correctly classified
    comparison_ert = 1*(y_predicted_ert == y_test)
    comparison_rf = 1*(y_predicted_rf == y_test)
    comparison_xgb = 1*(y_predicted_xgb == y_test)

    # percentage of correctly classified
    percentage_ert = np.zeros(num_test_patients)
    percentage_rf = np.zeros(num_test_patients)
    percentage_xgb = np.zeros(num_test_patients)
    for i in range(num_test_patients):
        percentage_ert[i] = sum(comparison_ert[i*100:(i+1)*100]) / num_samples
        percentage_rf[i] = sum(comparison_rf[i*100:(i+1)*100]) / num_samples
        percentage_xgb[i] = sum(comparison_xgb[i*100:(i+1)*100]) / num_samples

    return percentage_ert, percentage_rf, percentage_xgb


num_experiments = 100
percentage_ert, percentage_rf, percentage_xgb = \
    np.zeros(len(patients_array)), np.zeros(len(patients_array)), np.zeros(len(patients_array))
num_updates = np.zeros(len(patients_array))
num_models = len(patients_array)
for i in tqdm(range(num_experiments)):
    patients_array_train, patients_array_test = train_test_split(patients_array, test_size=0.33)
    x_train, y_train = generate_data(patients_array_train, shuffle_flag=True)
    x_test, y_test = generate_data(patients_array_test, shuffle_flag=False)

    percentage_ert_v, percentage_rf_v, percentage_xgb_v = \
        train_and_evaluate_ml_models(x_train, y_train, x_test, y_test,
                                     num_test_patients=len(patients_array_test), num_samples=100)

    for patient, i in zip(patients_array_test, range(len(patients_array_test))):  # patients_array_test:
        index = np.argwhere(patients_array==patient)[0][0]
        num_updates[index] += 1
        percentage_ert[index] += percentage_ert_v[i]
        percentage_rf[index] += percentage_rf_v[i]
        percentage_xgb[index] += percentage_xgb_v[i]

if not 0 in num_updates:
    print("For ERT for parties ", patients_array, ":")
    print(percentage_ert / num_updates)
    print("\nFor RF for parties ", patients_array, ":")
    print(percentage_rf / num_updates)
    print("\nFor XGB for parties ", patients_array, ":")
    print(percentage_xgb / num_updates)

    np.savetxt("1in5.csv",
               np.array([percentage_ert / num_updates, percentage_rf / num_updates, percentage_xgb / num_updates]),
               delimiter=",")

