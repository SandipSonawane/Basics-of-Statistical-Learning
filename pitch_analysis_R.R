# import libraries
library(tidyverse)
library(corrplot)
library(readr)
library(ggplot2)
library(skimr)
library(caret)
library(MLmetrics)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(pROC)
library(gridExtra)

# read subset of data
pitches_2019_regular_04 = readr::read_csv("data/pitches-2019-regular-04.csv")
pitches_2019_regular_05 = readr::read_csv("data/pitches-2019-regular-05.csv")
pitches_2019_regular_06 = readr::read_csv("data/pitches-2019-regular-06.csv")
pitches_2019_regular_07 = readr::read_csv("data/pitches-2019-regular-07.csv")
pitches_2019_regular_08 = readr::read_csv("data/pitches-2019-regular-08.csv")
pitches_2019_regular_09 = readr::read_csv("data/pitches-2019-regular-09.csv")
pitches_2019_post = readr::read_csv("data/pitches-2019-post.csv")

# merge regular season data
pitches_2019_regular = dplyr::bind_rows(
  pitches_2019_regular_04,
  pitches_2019_regular_05,
  pitches_2019_regular_06,
  pitches_2019_regular_07,
  pitches_2019_regular_08,
  pitches_2019_regular_09
)

names(pitches_2019_regular)
skimr::skim(pitches_2019_regular)
head(pitches_2019_regular)


# loooking into data
# the response variable is pitch_type
df_raw = pitches_2019_regular
df_raw$pitch_type = factor(df_raw$pitch_type)
summary(df_raw$pitch_type)
# there is a significant label imbalance here. Pitch type FF has large number of records and pitch type FS has least number of recrds.
# batter information is not going to have any effect on our response variable, hence these can be safely ignored.
# Even though pitcher information might be correlated with response variable, we want to generalize the model, hence we will ignore this variable as well.
# game_date does not have any relation with pitch_type, hence we will ignore this variable as well.
# better and pitcher are just some ids associated with batter name and pitcher name so these will be ignored as well.
# rest all numeric variables seem to be important.
# release spin rate has 12496 missing values, lets see its distribution

df_raw = df_raw[!is.na(df_raw$ax),]

summary(df_raw$ax)
skimr::skim(df_raw)

df_raw = as_tibble(df_raw)

df = df_raw%>%select(pitch_type, pitch_type, release_speed, release_pos_x, release_pos_y,
            release_pos_z, pfx_x, pfx_z, plate_x, plate_z, vx0, vy0, vz0, ax, ay,              
            az, effective_speed, release_spin_rate, release_extension)

df = drop_na(df)

df$pitch_type = factor(df$pitch_type)


# Look at each variable and check if there is any missing data.
mean(is.na(df))
column_NAs = function(x){
  sum(is.na(df[,x]))
}
NAs = sapply(1:ncol(df), column_NAs)
NAs
nrow(df)

# release_spin_rate has 12144 values of NA out of 709600. But we will ignore since its less than 30% of missing data.

df_test = df[1:700000,]

correlation_matrix = cor(df[,2:length(df)], use = "na.or.complete")

corrplot(correlation_matrix , order = "hclust", type = 'upper',
         tl.col = "slategray", tl.srt = 45, tl.cex = 0.65)

# pfx_z is highly negatively correlated with release_pos_y, hence we will drop pfx_z
# vx0 is also highly correlated with release position x, hence we will dorp vx0
# we will also drop vy0 since it is highly correlated with release position y
# release_position_y - release_extension is also correlated, lets drop release extension
# pfx_x-ax correlated as well, drop pfx_x
summary(df$effective_speed)
summary(df$release_speed)

# there is a minor difference in speed distribution of effective speed and release speed, so lets keep both for now.

df = df_raw%>%select(pitch_type, pitch_type, release_speed, release_pos_x, release_pos_y,
                     release_pos_z, plate_x, plate_z, vz0, ax, ay,              
                     az, effective_speed, release_spin_rate)

df$pitch_type = factor(df$pitch_type)

correlation_matrix = cor(df[,2:length(df)], use = "na.or.complete")

corrplot(correlation_matrix , order = "hclust", type = 'upper',
         tl.col = "slategray", tl.srt = 45, tl.cex = 0.65)


# lets see if release speed is different for different pitches

p1 = ggplot(df, aes(x=1, y=release_speed, fill=pitch_type)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(20, 28, by = 1))+
  # coord_cartesian(ylim = c(-20, 15))+
  
  labs(title='Release speed by pitch type', x='', y = 'Release Speed')+
  geom_boxplot(outlier.shape = NA)

names(df)

p2 = ggplot(df, aes(x=1, y=release_spin_rate, fill=pitch_type)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(20, 28, by = 1))+
  # coord_cartesian(ylim = c(-20, 15))+
  
  labs(title='release_spin_rate by pitch type', x='', y = 'release_spin_rate')+
  geom_boxplot(outlier.shape = NA)


p3 = ggplot(df, aes(x=1, y=release_pos_x, fill=pitch_type)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(20, 28, by = 1))+
  # coord_cartesian(ylim = c(-20, 15))+
  
  labs(title='release_pos_x by pitch type', x='', y = 'release_pos_x')+
  geom_boxplot(outlier.shape = NA)


p4 = ggplot(df, aes(x=1, y=release_pos_y, fill=pitch_type)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(20, 28, by = 1))+
  # coord_cartesian(ylim = c(-20, 15))+
  
  labs(title='release_pos_y by pitch type', x='', y = 'release_pos_y')+
  geom_boxplot(outlier.shape = NA)


p5 = ggplot(df, aes(x=1, y=release_pos_z, fill=pitch_type)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(20, 28, by = 1))+
  # coord_cartesian(ylim = c(-20, 15))+
  
  labs(title='release_pos_z by pitch type', x='', y = 'release_pos_z')+
  geom_boxplot(outlier.shape = NA)


p6 = ggplot(df, aes(x=1, y=ax, fill=pitch_type)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(20, 28, by = 1))+
  # coord_cartesian(ylim = c(-20, 15))+
  
  labs(title='ax by pitch type', x='', y = 'ax')+
  geom_boxplot(outlier.shape = NA)

gridExtra::grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3)

###### Model Building

# 1. Decision tree Model


# split the data with stratification
set.seed(42)
class_idx = createDataPartition(df$pitch_type, p = 0.8, list = FALSE)
class_trn = df[class_idx, ]
class_tst = df[-class_idx, ]

model_Control = trainControl(method = "cv", number = 5, classProbs=T,
                             savePredictions = T, verboseIter = F)

decision_tree_model1 = train(
  form = pitch_type ~ .,
  data = class_trn,
  trControl = model_Control,
  method = "rpart",
  na.action = na.pass
)

decision_tree_model = decision_tree_model1

# predictedOnTraining = decision_tree_model$pred$pred

predictedOnTesting = predict(decision_tree_model, class_tst)

# plot confusion matrix
confMatrix = confusionMatrix(class_tst$pitch_type, predictedOnTesting)
confMatrix$table
row_names = row.names(confMatrix$table)
column_names = rev(row_names)
confTable = confMatrix$byClass[,c(1,2,5,6,7)]
confTable

ggplot(as.data.frame(confMatrix$table), aes(Prediction,sort(Reference,decreasing = T), fill= Freq)) +
  geom_tile() +
  geom_text(aes(label=Freq)) +
  theme(text = element_text(size=16))+
  scale_fill_gradient(low="#ffffff", high="#009273") +
  labs(x = "Actual",y = "Predicted") +
  scale_x_discrete(labels=row_names) +
  scale_y_discrete(labels=column_names)



# 2. xgBoost Model

trn = as.matrix(class_trn[,2:length(df)])
tst = as.matrix(class_tst[,2:length(df)])
y = as.numeric(class_trn$pitch_type)-1
ytst = as.numeric(class_tst$pitch_type)-1

model_Control = trainControl(method = "cv", number = 5, summaryFunction=prSummary,
                             classProbs=T, savePredictions = T, verboseIter = F)

bst <- xgboost(data = trn, label = y, trControl = model_Control,
               max_depth = 3, eta = 1, nthread = 2, nrounds = 100,
               objective = "multi:softmax",  num_class=9, metric = 'accuracy')


bst$evaluation_log
predictedOnTesting = predict(bst, tst)
ytst

# plot confusion matrix
confMatrix = confusionMatrix(factor(ytst), factor(predictedOnTesting))
confMatrix$table

confTable = confMatrix$byClass[,c(1,2,5,6,7)]
row.names(confTable) = row_names
confTable

write.csv(confTable,'xgBoostResults.csv')

# row_names = row.names(confMatrix$table)
# column_names = rev(row_names)

ggplot(as.data.frame(confMatrix$table), aes(Prediction,sort(Reference,decreasing = T), fill= Freq)) +
  geom_tile() +
  geom_text(aes(label=Freq)) +
  theme(text = element_text(size=16))+
  scale_fill_gradient(low="#ffffff", high="#009273") +
  labs(x = "Actual",y = "Predicted") +
  scale_x_discrete(labels=row_names) +
  scale_y_discrete(labels=column_names)

















