####### 1.Data Preparation

# import data
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


df = read_csv("creditcard.csv")

head(df,5)

# The response variable 'Class' is of type double, We need to change it to factor.
df$Class = factor(ifelse(df$Class == 0, "genuine", "fraud"))

str(df)
# The data is in correct format. The PCA transformations have correct variable type.

# Look at each variable and check if there is any missing data.
mean(is.na(df))
column_NAs = function(x){
  sum(is.na(df[,x]))
}
NAs = sapply(1:ncol(df), column_NAs)
NAs

# As we can see from the above output, there are no missing data from this dataset.

# Data looks clean enough to proceed to next step

####### 2. EDA

skim(df)

cor(df[,1:30])

corrplot(cor(df[,1:30]), order = "hclust", type = 'upper',
         tl.col = "slategray", tl.srt = 45, tl.cex = 0.65)

# ![an image caption Source: Ultimate Funny Dog Videos Compilation 2013.](corr_plot.png){width=50%}
# We can see from the correlation plot that the principle components have zero correlation with each other.

# compare all the variables with box plots.


names(df)

# Change box plot colors by groups

# View for V1 ro V10
ggplot(df, aes(x=1, y=V1, fill=Class)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(1, 10, by = 1))+
  coord_cartesian(ylim = c(-20, 15))+
  labs(title='Boxplot Var 1 to 10', x='Variable Number', y = 'Variable Value')+
  geom_boxplot(outlier.shape = NA)+
  geom_boxplot(aes(x=2, y=V2, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=3, y=V3, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=4, y=V4, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=5, y=V5, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=6, y=V6, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=7, y=V7, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=8, y=V8, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=9, y=V9, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=10, y=V10, fill=Class), outlier.shape = NA)
  
# View for V11 ro V19
ggplot(df, aes(x=11, y=V11, fill=Class)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(11, 19, by = 1))+
  coord_cartesian(ylim = c(-20, 15))+
  labs(title='Boxplot Var 11 to 19', x='Variable Number', y = 'Variable Value')+
  geom_boxplot(outlier.shape = NA)+
  geom_boxplot(aes(x=12, y=V12, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=13, y=V13, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=14, y=V14, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=15, y=V15, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=16, y=V16, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=17, y=V17, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=18, y=V18, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=19, y=V19, fill=Class), outlier.shape = NA)

# View for 20 ro V28
ggplot(df, aes(x=20, y=V20, fill=Class)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  scale_x_continuous(breaks = seq(20, 28, by = 1))+
  coord_cartesian(ylim = c(-20, 15))+
  labs(title='Boxplot Var 20 to 28', x='Variable Number', y = 'Variable Value')+
  geom_boxplot(outlier.shape = NA)+
  geom_boxplot(aes(x=21, y=V21, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=22, y=V22, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=23, y=V23, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=24, y=V24, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=25, y=V25, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=26, y=V26, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=27, y=V27, fill=Class), outlier.shape = NA)+
  geom_boxplot(aes(x=28, y=V28, fill=Class), outlier.shape = NA)


summary(df$Time)
summary(df$Amount)

ggplot(df, aes(x=1, y=log(Amount), fill=Class)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  # scale_x_continuous(breaks = seq(20, 28, by = 1))+
  coord_cartesian(ylim = c(-5, 10))+
  labs(title='Transaction Amount by Class', x='', y = 'log(amount')+
  geom_boxplot(outlier.shape = NA)


df$Time

ggplot(df, aes(x=Time, y=Class, color = Class)) +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  labs(title='Transactions type with time', x='', y = 'Transaction Type')+
  geom_point(size = 0.5)


###### Model Building

# 1. Logistic Regression Model


# split the data with stratification

set.seed(42)
class_idx = createDataPartition(df$Class, p = 0.8, list = FALSE)
class_trn = df[class_idx, ]
class_tst = df[-class_idx, ]

model_Control = trainControl(method = "cv", number = 5, summaryFunction=prSummary,
                           classProbs=T, savePredictions = T, verboseIter = F)

logistic_model = train(
  form = Class ~ .-Time-V20-V22-V23-V24-V25-V26-V27,
  data = class_trn,
  trControl = model_Control,
  method = "glm",
  family = "binomial",
  metric = "AUC"
)

logistic_model


# get ROC data
predictions_for_test_logistic = predict(logistic_model, class_tst, type = 'prob')
predictions_prob_for_test_logistic = predictions_for_test_logistic[,1]


ytst = ifelse(class_tst$Class=='genuine', 0, 1)
ytst = as.matrix(ytst)

roc_test_logistic <- roc(ytst, predictions_prob_for_test_logistic, algorithm = 2)
roc_test_logistic
mean(roc_test_logistic$sensitivities)
mean(roc_test_logistic$specificities)


# 2. Decision Trees Model


model_Control = trainControl(method = "cv", number = 5, summaryFunction=prSummary,
                             classProbs=T, savePredictions = T, verboseIter = F)

decision_tree_model = train(
  form = Class ~ .-Time-V20-V22-V23-V24-V25-V26-V27,
  data = class_trn,
  trControl = model_Control,
  method = "rpart",
  metric = "AUC",
  tuneGrid = expand.grid(cp = c(0,0.1,0.01,0.001))
)

decision_tree_model


# get ROC data
predictions_for_test_decision_tree = predict(decision_tree_model, class_tst, type = 'prob')
predictions_prob_for_test_decision_tree = predictions_for_test_decision_tree[,1]

ytst = ifelse(class_tst$Class=='genuine', 0, 1)
ytst = as.matrix(ytst)

roc_test_decision_tree <- roc(ytst, predictions_prob_for_test_decision_tree, algorithm = 2)
roc_test_decision_tree
mean(roc_test_decision_tree$specificities)


# knn model


knn_model = knn3(Class ~ .-Time-V20-V22-V23-V24-V25-V26-V27, data = class_trn, k = 10)
predictions_for_test_knn = predict(knn_model, class_tst, type = 'prob')
predictions_prob_for_test_knn = predictions_for_test_knn[,1] 

min(predictions_prob_for_test_knn[,1])



ytst = ifelse(class_tst$Class=='genuine', 0, 1)
ytst = as.matrix(ytst)

roc_test_knn <- roc(ytst, predictions_prob_for_test_knn, algorithm = 2)
roc_test_knn
mean(roc_test_knn$sensitivities)
mean(roc_test_knn$specificities)


# xgboost model

class_idx = createDataPartition(df$Class, p = 0.8, list = FALSE)
class_trn = df[class_idx, ]
class_tst = df[-class_idx, ]

trn = as.matrix(class_trn[,1:22])
y = ifelse(class_trn$Class=='genuine', 0, 1)

model_Control = trainControl(method = "cv", number = 5, summaryFunction=prSummary,
                             classProbs=T, savePredictions = T, verboseIter = F)

bst <- xgboost(data = trn, label = y, trControl = model_Control,
               max_depth = 3, eta = 1, nthread = 2, nrounds = 100,
               objective = "binary:logistic", eval_metric = "auc")



# bst <- xgboost(data = trn, label = y,
#                max_depth = 3, eta = 1, nthread = 2, nrounds = 100,
#                objective = "binary:logistic", eval_metric = "auc")

# bst$evaluation_log


tst = as.matrix(class_tst[,1:22])

predictions_for_test_xgb = predict(bst, tst, type = 'prob')
length(predictions_for_test_xgb)


ytst = ifelse(class_tst$Class=='genuine', 0, 1)
ytst = as.matrix(ytst)

roc_test_xgb <- roc(ytst, predictions_for_test_xgb, algorithm = 2)
roc_test_xgb
head(round(roc_test_xgb$specificities, 4),80)
mean((roc_test_xgb$sensitivities))
mean(roc_test_xgb$specificities)
median(roc_test_xgb$sensitivities)
length(roc_test_xgb$sensitivities)

# plot ROC curve

df_roc = data.frame()
roc_test_decision_tree
roc_test_knn
roc_test_xgb


lineThichness = 1
pltColors = c('Logistic Regression' = 'red', 'Decision Tree' = 'blue', 'knn' = 'green', 'Xgboost' = 'orange')

ggplot() +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  
  labs(title='ROC Curve', x='1 - Specificity (False Positive Rate)', y = 'Sensitivity (True Positive Rate)')+
  
  geom_line(aes(y = roc_test_logistic$sensitivities, x = 1- roc_test_logistic$specificities, color = 'Logistic Regression'),
            size = lineThichness)+
  
  geom_line(aes(y = roc_test_decision_tree$sensitivities, x = 1-roc_test_decision_tree$specificities, color = 'Decision Tree'),
            size = lineThichness)+
  
  geom_line(aes(y = roc_test_knn$sensitivities, x = 1-roc_test_knn$specificities, color = 'knn'),
            size = lineThichness)+
  
  geom_line(aes(y = roc_test_xgb$sensitivities, x = 1-roc_test_xgb$specificities, color = 'Xgboost'),
            size = lineThichness)+
  
  scale_colour_manual(values=pltColors)




# plot ROC cuver for XGBoost and select best threshold
## check what threshold represent, does it represent actual probability?

roc_test_xgb$thresholds

prob_threshold = roc_test_xgb$thresholds

ggplot() +
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme_set(theme_bw())+
  theme(text = element_text(size=16))+
  
  labs(title='ROC Curve', x='1 - Specificity (False Positive Rate)', y = 'Sensitivity (True Positive Rate)')+
  
  geom_line(aes(y = roc_test_xgb$sensitivities, x = 1-roc_test_xgb$specificities,
                color = prob_threshold), size = lineThichness)+
  
  scale_color_gradient(low="blue", high="red")
  
