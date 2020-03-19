import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

seed = 42

# https://github.com/sayibet/fight-detection-surv-dataset
DATASET_SURV = "/home/andrei/Desktop/Datasets/fight-detection-surv-dataset"
FIGHT_SURV_DIR = "{}/fight/*".format(DATASET_SURV)
NOFIGHT_SURV_DIR = "{}/noFight/*".format(DATASET_SURV)

# My private dataset recorded in the offices, can't be shared
DATASET_OFFICE = "/home/andrei/Desktop/Datasets/office-fights"
FIGHT_OFFICE_DIR = "{}/fight/**/*.avi".format(DATASET_OFFICE)
NOFIGHT_OFFICE_DIR = "{}/no-fight/**/*.avi".format(DATASET_OFFICE)

# combination of several open datasets
DATASET_SURV_EXT = "/home/andrei/Desktop/Datasets/fight-detection-surv-ext"
FIGHT_SURV_EXT_DIR = r"{}/fight/**/*.[ma][pv][4i]".format(DATASET_SURV_EXT)
NOFIGHT_SURV_EXT_DIR = r"{}/noFight/**/*.[ma][pv][4ig]".format(DATASET_SURV_EXT)

# http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89
DATASET_HOCKEY = "/home/andrei/Desktop/Datasets/HockeyFights"
FIGHT_HOCKEY_DIR = "{}/fight/**/*.avi".format(DATASET_HOCKEY)
NOFIGHT_HOCKEY_DIR = "{}/noFight/**/*.avi".format(DATASET_HOCKEY)

# http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635/tech&hit=1&filelist=1
DATASET_MOVIES = "/home/andrei/Desktop/Datasets/Peliculas"
FIGHT_MOVIES_DIR = "{}/fights/**/*.avi".format(DATASET_MOVIES)
NOFIGHT_MOVIES_DIR = "{}/noFights/**/*.mpg".format(DATASET_MOVIES)


def collect(fight_path, nofight_path):
   fight_files = np.array(glob(fight_path, recursive=True))
   fight = np.column_stack((fight_files, np.full(fight_files.shape, 1)))
   nofight_files = np.array(glob(nofight_path, recursive=True))
   nofight = np.column_stack((nofight_files, np.full(nofight_files.shape, 0)))
   data = np.vstack((fight, nofight))
   return data

def sample_data(data, ratio):
    classes = data[:,1]
    class_0_ix = np.where(classes == '0')[0]
    class_1 = data[classes == '1', :]
    # we always have less fights then no fights
    np.random.seed(42)
    class_0_sample_ix = np.random.choice(class_0_ix, class_1.shape[0]*ratio, )
    data = np.vstack((data[class_0_sample_ix,:], class_1))
    return data[:,0], data[:,1].astype(np.int8)

def surv_experiment():
    data = collect(FIGHT_SURV_DIR, NOFIGHT_SURV_DIR)
    return  train_test_split(data[:,0], data[:,1].astype(np.int8), test_size=0.33, random_state=42)

def surv_ext_experiment():
    data = collect(FIGHT_SURV_EXT_DIR, NOFIGHT_SURV_EXT_DIR)
    return  train_test_split(data[:,0], data[:,1].astype(np.int8), test_size=0.33, random_state=42)

# validation and testing on one dataset
def surv_ext_experiment_complete():
    data = collect(FIGHT_SURV_EXT_DIR, NOFIGHT_SURV_EXT_DIR)
    X = data[:,0]
    y = data[:,1].astype(np.int8)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, stratify=y_rest, test_size=0.2, random_state=42)
    return  X_train, X_valid, X_test, y_train, y_valid, y_test 

def office_test(sample=True, ratio=2):
    data = collect(FIGHT_OFFICE_DIR, NOFIGHT_OFFICE_DIR)
    if sample:
        return sample_data(data, ratio)
    return  data[:,0], data[:,1].astype(np.int8)

def hockey_test():
    data = collect(FIGHT_HOCKEY_DIR, NOFIGHT_HOCKEY_DIR)
    return  data[:,0], data[:,1].astype(np.int8)

def movies_test():
    data = collect(FIGHT_MOVIES_DIR, NOFIGHT_MOVIES_DIR)
    return  data[:,0], data[:,1].astype(np.int8)
