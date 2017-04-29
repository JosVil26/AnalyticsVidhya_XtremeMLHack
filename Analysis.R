set.seed(260582)
## El mejor modelo fue el hecho con XGBoost usando solo fechas y categorías

 library(lubridate)
 library(caret)
 library(randomForest)
 library(Hmisc)
 library(data.table)
 library(dplyr)
 library(xgboost)

setwd("~/Coursera/Data Scientist Specialization/R Workspace/Xtreme ML Hack")
Contacts_Train <- read.table("./Train/Contacts_Pre_2017.csv", sep = ",", header = T, stringsAsFactors = F, na.strings=c("NA","#DIV/0!",""))
# Contracts_End <- read.table("./Train/Contracts_End.csv", sep = ",", header = T, stringsAsFactors = F, na.strings=c("NA","#DIV/0!",""))
# Contracts_New <- read.table("./Train/Contracts_New.csv", sep = ",", header = T, stringsAsFactors = F, na.strings=c("NA","#DIV/0!",""))
Resolution_Train <- read.table("./Train/Resolution_Pre_2017.csv", sep = ",", header = T, stringsAsFactors = F, na.strings=c("NA","#DIV/0!",""))
Contacts_Test <- read.table("./Test/Contacts2017.csv", sep = ",", header = T, stringsAsFactors = F, na.strings=c("NA","#DIV/0!",""))
Resolution_Test <- read.table("./Test/Resolution2017.csv", sep = ",", header = T, stringsAsFactors = F, na.strings=c("NA","#DIV/0!",""))

null_values <- names(Resolution_Train) %in% c("End_Date", "City", "Status", 
                                              "Theme", "Reason", "Water.Usage", 
                                              "Postal.code")

Resolution_Train <- Resolution_Train[!null_values]

null_values <- names(Contacts_Train) %in% c("END.DATE")

Contacts_Train <- Contacts_Train[!null_values]

Sys.setlocale("LC_TIME", "C")
Contacts_Train$START.DATE <- as.Date(Contacts_Train$START.DATE, "%Y-%m-%d")
Contacts_Train$CONTACT.TYPE <- as.factor(Contacts_Train$CONTACT.TYPE)
# Contracts_End$MUNICIPALITY <- as.factor(Contracts_End$MUNICIPALITY)
# Contracts_End$WATER_USAGE <- as.factor(Contracts_End$WATER_USAGE)
# Contracts_End$TYPE_OF_HOUSEHOLD <- as.factor(Contracts_End$TYPE_OF_HOUSEHOLD)
# Contracts_End$DATE_END_CONTRACT <- as.Date(paste(Contracts_End$YEAR_END_CONTRACT, "-", Contracts_End$MONTH_END_CONTRACT, "-", Contracts_End$DAY_END_CONTRACT, sep = ""), "%Y-%b.-%d")
# Contracts_End$MONTH_END_CONTRACT <- month(Contracts_End$DATE_END_CONTRACT)
# Contracts_New$MUNICIPALITY <- as.factor(Contracts_New$MUNICIPALITY)
# Contracts_New$WATER_USAGE <- as.factor(Contracts_New$WATER_USAGE)
# Contracts_New$TYPE_OF_HOUSEHOLD <- as.factor(Contracts_New$TYPE_OF_HOUSEHOLD)
# Contracts_New$DATE_ALTA_CONTRACT <- as.Date(paste(Contracts_New$YEAR_CONTRACT, "-", Contracts_New$MONTH_CONTRACT, "-", Contracts_New$DAY_ALTA_CONTR, sep = ""), "%Y-%b.-%d")
# Contracts_New$MONTH_CONTRACT <- month(Contracts_New$DATE_ALTA_CONTRACT)
Resolution_Train$Date <- as.Date(Resolution_Train$Date)
# Resolution_Train$End_Date <- as.Date(Resolution_Train$End_Date)
# Resolution_Train$City <- as.factor(Resolution_Train$City)
# Resolution_Train$Status <- as.factor(Resolution_Train$Status)
Resolution_Train$Category <- as.factor(Resolution_Train$Category)
Resolution_Train$Subject <- as.factor(Resolution_Train$Subject)
# Resolution_Train$Theme <- as.factor(Resolution_Train$Theme)
# Resolution_Train$Reason <- as.factor(Resolution_Train$Reason)
# Resolution_Train$Water.Usage <- as.factor(Resolution_Train$Water.Usage)
Contacts_Test$Date <- as.Date(Contacts_Test$Date, "%Y-%m-%d")
Contacts_Test$Contacts <- as.integer(Contacts_Test$Contacts)
Contacts_Test$CONTACT.TYPE <- as.factor(Contacts_Test$CONTACT.TYPE)

# CONTACTS CASE
Contacts_Train <- Contacts_Train %>% rename(Date = START.DATE)
Contacts_Train <- Contacts_Train %>% group_by(Date, CONTACT.TYPE) %>%
        summarise(Contacts = sum(Contacts, na.rm = T))
# Q3_Contacts <- quantile(Contacts_Train$Contacts)[4]
# Q1_Contacts <- quantile(Contacts_Train$Contacts)[2]
# IQR_Contacts <- IQR(Contacts_Train$Contacts)
# Contacts_Train <- Contacts_Train %>% filter(Contacts <= Q3_Contacts + 1.5 * IQR_Contacts & 
#                                                     Contacts >= Q1_Contacts - 1.5 * IQR_Contacts)
Contacts_Train <- Contacts_Train %>% mutate(Year = year(Date), 
                                            Month = month(Date), Day = day(Date)) %>%
        select(Date, Year, Month, Day, CONTACT.TYPE, Contacts)
Contacts_Train <- as.data.frame(Contacts_Train)
Contacts_Train$Year <- as.numeric(Contacts_Train$Year)
Contacts_Train$Month <- as.numeric(Contacts_Train$Month)
Contacts_Train$Day <- as.numeric(Contacts_Train$Day)
Contacts_Train$DayYear <- as.numeric(yearDays(Contacts_Train$Date))
Contacts_Train$DayMonth <- as.numeric(monthDays(Contacts_Train$Date))
Contacts_Train$DayWeek <- weekdays(Contacts_Train$Date)
Contacts_Train$Quarters <- quarter(Contacts_Train$Date)
cor(Contacts_Train[,c(2:4,6:8,10)])
pcaData <- princomp(Contacts_Train[,c(2:4,6:8,10)], scores = T, cor = T)
summary(pcaData)
loadings(pcaData)
screeplot(pcaData, type = 'line', main = 'Screeplot')
biplot(pcaData)
pcaData$scores[1:10,]

Contacts_Test <- Contacts_Test %>% mutate(Year = year(Date), 
                                            Month = month(Date), Day = day(Date)) %>%
        select(Date, Year, Month, Day, CONTACT.TYPE, Contacts, ID)
Contacts_Test <- as.data.frame(Contacts_Test)
Contacts_Test$Year <- as.numeric(Contacts_Test$Year)
Contacts_Test$Month <- as.numeric(Contacts_Test$Month)
Contacts_Test$Day <- as.numeric(Contacts_Test$Day)

test_ids <- Contacts_Test$ID
target <- Contacts_Train$Contacts


## XGBoost

xgtrain <- xgb.DMatrix(data = data.matrix(Contacts_Train[,c(1,5)]), label = target, missing = NA)
xgtest <- xgb.DMatrix(data = data.matrix(Contacts_Test[,c(1,5)]), missing = NA)

class(xgtrain)
class(xgtest)
params <- list()
params$objective <- "reg:linear"
params$eta <- 0.1
params$max_depth <- 5
params$subsample <- 0.9
params$colsample_bytree <- 0.9
params$min_child_weight <- 10
params$eval_metric <- "rmse"

# cv
model_xgb_cv <- xgb.cv(params = params, xgtrain, nfold = 10, nrounds = 1000)

# training
model_xgb <- xgb.train(params = params, xgtrain, nrounds = 1000)

# prediction
pred <- predict(model_xgb, xgtest)

# submission
submit <- data.table(ID = test_ids, Contacts = pred)
write.csv(submit, file = "Contacts.csv", sep = ",", row.names = FALSE)

# RESOLUTION CASE
Resolution_Train <- Resolution_Train %>% group_by(Date, Category, Subject) %>%
        summarise(Resolution = sum(Resolution, na.rm = T))
Q3_Resolution <- quantile(Resolution_Train$Resolution)[4]
Q1_Resolution <- quantile(Resolution_Train$Resolution)[2]
IQR_Resolution <- IQR(Resolution_Train$Resolution)
Resolution_Train <- Resolution_Train %>% filter(Resolution <= Q3_Resolution + 1.5 * IQR_Resolution 
                                                & Resolution >= Q1_Resolution - 1.5 * IQR_Resolution)
Resolution_Train <- Resolution_Train %>% mutate(Year = year(Date), 
                                                Month = month(Date), Day = day(Date)) %>%
        select(Date, Year, Month, Day, Category, Subject, Resolution)

Resolution_Test$Date <- as.Date(Resolution_Test$Date)
Resolution_Test$Category <- as.factor(Resolution_Test$Category)
Resolution_Test$Subject <- as.factor(Resolution_Test$Subject)
Resolution_Test$Resolution <- as.integer(Resolution_Test$Resolution)

Resolution_Test <- Resolution_Test %>% mutate(Year = year(Date), 
                                              Month = month(Date), Day = day(Date)) %>%
        select(Date, Year, Month, Day, Category, Subject, Resolution, ID)

# Resolution_Train_Test <- rbind(Resolution_Train, Resolution_Test, use.names = T, fill = T)
# 
# Cat <- Resolution_Train_Test %>% group_by(Category) %>% summarise(count_Category = count(Category)) item <- Mlware_Train_Test[, .(count_item = .N), .(itemId)]
# 
# Mlware_Train <- merge(Mlware_Train, user, by = "userId")
# Mlware_Test <- merge(Mlware_Test, user, by = "userId")
# 
# Mlware_Train <- merge(Mlware_Train, item, by = "itemId")
# Mlware_Test <- merge(Mlware_Test, item, by = "itemId")
# 
# Mlware_Train$count_user <- as.numeric(Mlware_Train$count_user)
# Mlware_Train$count_item <- as.numeric(Mlware_Train$count_item)
# 
# Mlware_Test$count_user <- as.numeric(Mlware_Test$count_user)
# Mlware_Test$count_item <- as.numeric(Mlware_Test$count_item)
# 
test_ids <- Resolution_Test$ID
target <- Resolution_Train$Resolution


## XGBoost

xgtrain <- xgb.DMatrix(data = data.matrix(Resolution_Train[,c(1,5,6)]), label = target, missing = NA)
xgtest <- xgb.DMatrix(data = data.matrix(Resolution_Test[,c(1,5,6)]), missing = NA)

class(xgtrain)
class(xgtest)
params <- list()
params$objective <- "reg:linear"
params$eta <- 0.1
params$max_depth <- 5
params$subsample <- 0.9
params$colsample_bytree <- 0.9
params$min_child_weight <- 10
params$eval_metric <- "rmse"

# cv
model_xgb_cv <- xgb.cv(params = params, xgtrain, nfold = 10, nrounds = 1000)

# training
model_xgb <- xgb.train(params = params, xgtrain, nrounds = 1000)

# prediction
pred <- predict(model_xgb, xgtest)

# submission
submit <- data.table(ID = test_ids, Resolution = pred)
write.csv(submit, file = "Resolution.csv", sep = ",", row.names = FALSE)
# glm
# Mlware_Train[, ":="(ID = NULL)]
# Mlware_Test[, ":="(ID = NULL)]
# Mlware_Train <- as.data.frame(Mlware_Train)
# Mlware_Test <- as.data.frame(Mlware_Test)
inTrain <- createDataPartition(Resolution_Train$Resolution, p=0.7, list=FALSE)
subTraining <- Resolution_Train[inTrain, ]
subTesting <- Resolution_Train[-inTrain, ]


model <- glm(Resolution ~.,family=gaussian(link='identity'),data=subTraining)
rmse <- sqrt(mean((subTesting$Resolution-predict(model, newdata = subTesting))^2))
rmse

# Regression Tree fit the model after preprocessing 
modelFit1 <- train(Contacts ~., method="rpart", preProcess=c("center", "scale"), 
                   data=subTraining, trControl = trainControl(method="cv", 
                                                              number=10))
rmse1 <- sqrt(mean((subTesting$Contacts-predict(modelFit1, newdata = subTesting))^2))
rmse1

# Random Forest fit the model after preprocessing 
modelFit2 <- randomForest(Contacts ~., data=subTraining)
rmse2 <- sqrt(mean((subTesting$Contacts-predict(modelFit2, newdata = subTesting))^2))
rmse2

# Generalized Boosted Regression fit the model after preprocessing 
modelFit3 <- train(Contacts ~ ., data=subTraining, method = "gbm",
                   trControl = trainControl(method = "repeatedcv",
                                            number = 5,
                                            repeats = 1),
                   verbose = FALSE)
rmse3 <- sqrt(mean((subTesting$Contacts-predict(modelFit3, newdata = subTesting))^2))
rmse3

# Generalized Linear Regression fit the model after preprocessing 
modelFit4 <- train(Contacts ~., data = subTraining, method = "glm", family = "gaussian")
rmse4 <- sqrt(mean((subTesting$Contacts-predict(modelFit4, newdata = subTesting))^2))
rmse4

total1 <- c(result1$overall['Accuracy'], result1$byClass[1], result1$byClass[2], auc1, "rtree")
total2 <- c(result2$overall['Accuracy'], result2$byClass[1], result2$byClass[2], auc2, "RF")
total3 <- c(result3$overall['Accuracy'], result3$byClass[1], result3$byClass[2], auc3, "gbm")
total4 <- c(result4$overall['Accuracy'], result4$byClass[1], result4$byClass[2], auc4, "glm")
totalx <- c(result$overall['Accuracy'], result$byClass[1], result$byClass[2], auc, "lm")
total <- rbind(total1, total2, total3, total4, totalx)
total
write.csv(total, file = "Without_Null_NA_Total_80.csv", sep = ",")

predglm <- predict(model, newdata=Resolution_Test, type="response")
zglm = as.matrix(data.frame(test_ids, predglm)) 
colnames(zglm) <- c("ID", "Resolution")
write.csv(zglm, file = "Resolution.csv", sep = ",", row.names = FALSE)

pred1 <- as.numeric(as.vector(predict(modelFit1, newdata=Contacts_Test)))
z1 = as.matrix(data.frame(test_ids, pred1)) 
colnames(z1) <- c("ID", "Contacts")
write.csv(z1, file = "Contacts.csv", sep = ",", row.names = FALSE)

pred2 <- as.numeric(as.vector(predict(modelFit2, newdata=Contacts_Test)))
z2 = as.matrix(data.frame(test_ids, pred2)) 
colnames(z2) <- c("ID", "Contacts")
write.csv(z2, file = "Contacts.csv", sep = ",", row.names = FALSE)

pred3 <- as.numeric(as.vector(predict(modelFit3, newdata=Contacts_Test)))
z3 = as.matrix(data.frame(test_ids, pred3)) 
colnames(z3) <- c("ID", "Contacts")
write.csv(z3, file = "Contacts.csv", sep = ",", row.names = FALSE)

pred4 <- as.numeric(as.vector(predict(modelFit4, newdata=Contacts_Test)))
z4 = as.matrix(data.frame(test_ids, pred4)) 
colnames(z4) <- c("ID", "Contacts")
write.csv(z4, file = "Contacts.csv", sep = ",", row.names = FALSE)


#CONTACTS
# glm
# Mlware_Train[, ":="(ID = NULL)]
# Mlware_Test[, ":="(ID = NULL)]
# Mlware_Train <- as.data.frame(Mlware_Train)
# Mlware_Test <- as.data.frame(Mlware_Test)
inTrain <- createDataPartition(Contacts_Train$Contacts, p=0.7, list=FALSE)
subTraining <- Contacts_Train[inTrain, ]
subTesting <- Contacts_Train[-inTrain, ]


model <- glm(Contacts ~.,family=gaussian(link='identity'),data=subTraining)
rmse <- sqrt(mean((subTesting$Contacts-predict(model, newdata = subTesting))^2))
rmse

# Regression Tree fit the model after preprocessing 
modelFit1 <- train(Contacts ~., method="rpart", preProcess=c("center", "scale"), 
                   data=subTraining, trControl = trainControl(method="cv", 
                                                              number=10))
rmse1 <- sqrt(mean((subTesting$Contacts-predict(modelFit1, newdata = subTesting))^2))
rmse1

# Random Forest fit the model after preprocessing 
modelFit2 <- randomForest(Contacts ~., data=subTraining)
rmse2 <- sqrt(mean((subTesting$Contacts-predict(modelFit2, newdata = subTesting))^2))
rmse2

# Generalized Boosted Regression fit the model after preprocessing 
modelFit3 <- train(Contacts ~ ., data=subTraining, method = "gbm",
                   trControl = trainControl(method = "repeatedcv",
                                            number = 5,
                                            repeats = 1),
                   verbose = FALSE)
rmse3 <- sqrt(mean((subTesting$Contacts-predict(modelFit3, newdata = subTesting))^2))
rmse3

# Generalized Linear Regression fit the model after preprocessing 
modelFit4 <- train(Contacts ~., data = subTraining, method = "glm", family = "gaussian")
rmse4 <- sqrt(mean((subTesting$Contacts-predict(modelFit4, newdata = subTesting))^2))
rmse4

total1 <- c(result1$overall['Accuracy'], result1$byClass[1], result1$byClass[2], auc1, "rtree")
total2 <- c(result2$overall['Accuracy'], result2$byClass[1], result2$byClass[2], auc2, "RF")
total3 <- c(result3$overall['Accuracy'], result3$byClass[1], result3$byClass[2], auc3, "gbm")
total4 <- c(result4$overall['Accuracy'], result4$byClass[1], result4$byClass[2], auc4, "glm")
totalx <- c(result$overall['Accuracy'], result$byClass[1], result$byClass[2], auc, "lm")
total <- rbind(total1, total2, total3, total4, totalx)
total
write.csv(total, file = "Without_Null_NA_Total_80.csv", sep = ",")

predglm <- predict(model, newdata=Contacts_Test, type="response")
zglm = as.matrix(data.frame(test_ids, predglm)) 
colnames(zglm) <- c("ID", "Contacts")
write.csv(zglm, file = "Contacts.csv", sep = ",", row.names = FALSE)

pred1 <- as.numeric(as.vector(predict(modelFit1, newdata=Contacts_Test)))
z1 = as.matrix(data.frame(test_ids, pred1)) 
colnames(z1) <- c("ID", "Contacts")
write.csv(z1, file = "Contacts.csv", sep = ",", row.names = FALSE)

pred2 <- as.numeric(as.vector(predict(modelFit2, newdata=Contacts_Test)))
z2 = as.matrix(data.frame(test_ids, pred2)) 
colnames(z2) <- c("ID", "Contacts")
write.csv(z2, file = "Contacts.csv", sep = ",", row.names = FALSE)

pred3 <- as.numeric(as.vector(predict(modelFit3, newdata=Contacts_Test)))
z3 = as.matrix(data.frame(test_ids, pred3)) 
colnames(z3) <- c("ID", "Contacts")
write.csv(z3, file = "Contacts.csv", sep = ",", row.names = FALSE)

pred4 <- as.numeric(as.vector(predict(modelFit4, newdata=Contacts_Test)))
z4 = as.matrix(data.frame(test_ids, pred4)) 
colnames(z4) <- c("ID", "Contacts")
write.csv(z4, file = "Contacts.csv", sep = ",", row.names = FALSE)

