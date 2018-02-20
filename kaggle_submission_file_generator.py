############################################################################
# Script to generate kaggle submission file based on IdLookupTable.csv
# Checkout kaggle_facial_keypoint_recognition_submission.csv file for 
# format.
############################################################################

import pandas as pd

# Load csv files
ids = pd.read_csv("../data/IdLookupTable.csv")
keypoints = pd.read_csv("../test_prediction_net6.csv")

# Print shape of files loaded
print ids.shape, keypoints.shape

# Merge both the files based on ImageId and FeatureName
sub = pd.merge(left = ids, right=keypoints, how="left", left_on=["ImageId","FeatureName"], right_on=["ImageId","FeatureName"])

# Generate pixel value based on the [-1,1] normalized predictions
sub['Location']=sub['Location_y']*48+48

# Save RowId and location to csv file for submission.
sub[[0,6]].to_csv('keypoint_submission_v0.2.csv', index=False)
