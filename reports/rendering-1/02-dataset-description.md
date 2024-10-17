# Dataset Description

The dataset is from a study that collected data from young adults in the UK with type 1 diabetes, who used a continuous glucose monitor (CGM), an insulin pump and a smartwatch.
These devices collected blood glucose readings, insulin dosage, carbohydrate intake, and activity data. The data collected was aggregated to five-minute intervals and formatted
into samples. Each sample represents a point in time and includes the aggregated five-minute intervals from the previous six hours. The aim is to predict the blood glucose reading
an hour into the future, for each of these samples.

The training set takes samples from the first three months of study data from nine of the participants and includes the future blood glucose value. These training samples appear in
chronological order and overlap. The testing set takes samples from the remainder of the study period from fifteen of the participants (so unseen participants appear in the testing
set). These testing samples do not overlap and are in a random order to avoid data leakage.

Complexities to be aware of:

* this is medical data so there are missing values and noise in the data
* the participants did not all use the same device models (CGM, insulin pump and smartwatch) so there may be differences in the collection method of the data
* some participants in the test set do not appear in the training set

## File descriptions

* activities.txt - a list of activity names that appear in the activity-X:XX columns
* sample_submission.csv - a sample submission file in the correct format
* test.csv - the test set
* train.csv - the training set

## Columns

| #Column | Name   | Description                                                                                     | Type   | 
|---------|---------------|-------------------------------------------------------------------------------------------------|--------|
| 1       | id            | row id consisting of participant number and a count for that participant                        | string |
| 2       | p_num         | participant number                                                                              | string |
| 3       | time          | time of day in the format HH:MM:SS                                                              | string |
| 4-75    | bg-X:XX       | blood glucose reading in mmol/L, X:XX(H:MM) time in the past                                    | float  |
| 76-147  | insulin-X:XX  | total insulin dose received in units in the last 5 minutes, X:XX(H:MM) time in the past         | float  |
| 148-219 | carbs-X:XX    | total carbohydrate value consumed in grammes in the last 5 minutes, X:XX(H:MM) time in the past | float  |
| 220-291 | hr-X:XX       | mean heart rate in beats per minute in the last 5 minutes, X:XX(H:MM) time in the past          | float  |
| 292-363 | steps-X:XX    | total steps walked in the last 5 minutes, X:XX(H:MM) time in the past                           | float  |
| 364-435 | cals-X:XX     | total calories burnt in the last 5 minutes, X:XX(H:MM) time in the past                         | string |
| 436-507 | activity-X:XX | self-declared activity performed in the last 5 minutes, X:XX(H:MM) time in the past             | string |
| 508     | bg-X:XX+1     | blood glucose reading in mmol/L, X:XX+1(H:MM) time in the future, not provided in test.csv      | float  |
