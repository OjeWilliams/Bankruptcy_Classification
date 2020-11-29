rm(list = ls())    #delete objects
cat("\014")        #clear console
gc()              #does garbage collections and frees up RAM
install.packages("gridExtra")
install.packages("caret")
install.packages("pROC")
install.packages("ROCR")
install.packages("randomForest")
library(gridExtra)
library(ROCR)
library(caret)
library(pROC)
library(randomForest)  
library(tidyverse)
library(glmnet)
library(DescTools)
set.seed(1453)

#downloading data directly from github repository 
data <- read_csv('https://raw.githubusercontent.com/Jorgelopez1992/bankruptcy_classification/master/bankruptcy_Train.csv',
                 col_names = T)

# Change "class" into a factor as well as relable for readability
data$class <- ifelse(test = data$class == 0, yes = "Solvent", no = "Bankrupt")
data$class <- as.factor(data$class)

mat.data <- as.matrix(data)
mat.data

#info for splitting into train/test in loop
n        =    nrow(mat.data)
p        =    ncol(mat.data) - 1

n.train        =     floor(0.9*n)
n.test         =     n-n.train

M              =     50   #total loops 
num_thrs       =     100  #Number of threshholds for classification for logistic regression (0.01,0.02 etc.)


#empty vectors to store time values
rf.time<-c(rep(0,M))

# create vectors to store test/train AUC
rf_train_auc <- rf_test_auc <- c(rep(0,M))


for (m in c(1:M)) {
  #randomly sampling rows 
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  train.data       =     data[train,]
  test.data        =     data[test,]
 
  #### Count of all solvent i.e '0'
  tot.solvent <-length(which(train.data$class == "Solvent"))
  #### Count of all bankrupt i.e '1'
  tot.bankrupt <- length(which(train.data$class == "Bankrupt"))
  
  # Creating weight vector
  ws = tot.bankrupt/n.train
  wb = 1
  weight.vec <- c("Solvent"=ws, "Bankrupt"=wb)

    ######################################################################################  
  ###########################   random forest   ########################################
  ###################################################################################### 
  
  # This tunes the random forest and selects the best mtry ( # of vars available for splitting at each node)
  mtry <- tuneRF(train.data[1:64],train.data$class, ntreeTry=500,
                 stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
  best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
  #print(mtry)
  #print(best.m)

  
  start.time<-Sys.time()
  
  rf.model   =     randomForest(class~.,data = train.data, mtry = best.m, ntree=500,classwt=weight.vec, importance = TRUE, proximity = TRUE)
  
  end.time<-Sys.time()
  rf.time[m]<-rf.time[m]+(end.time-start.time)
  #print(rf.model)
  
  # Plotting the OOB error rates to estimate if the default 500 trees are optimal
  #oob.error.data <- data.frame(
   # Trees=rep(1:nrow(rf.model$err.rate), times=3),
    #Type=rep(c("OOB", "Solvent", "Bankrupt"), each=nrow(rf.model$err.rate)),
    #Error=c(rf.model$err.rate[,"OOB"], 
     #       rf.model$err.rate[,"Solvent"], 
      #      rf.model$err.rate[,"Bankrupt"]))
  
#  ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
 #   geom_line(aes(color=Type))
  
  # All error rates seem to stabilize at 500 trees or less... tried 1000 but nothing changed really.
  
  # Prediction
  rid.fit <- predict(rf.model,train.data)
  
  # Check prediction
  #head(rid.fit)
  # Check the train.data
  #head(train.data$class)
  # Seems to be accurate so far
  predictions = as.data.frame(predict(rf.model, train.data, type = "prob"))
  predictions$predicted <- names(predictions)[1:3][apply(predictions[,1:2], 1, which.max)]
  predictions$observed <- train.data$class
  head(predictions)
  
  # Train AUC
  # The first way to calculate training auc
  rf_p_train <- predict(rf.model, type="prob")[,2]
  rf_pr_train <- prediction(rf_p_train, train.data$class)
  r_auc_train1 <- performance(rf_pr_train, measure = "auc")@y.values[[1]]
  #cat("AUC =",r_auc_train1,"\n")
     
  # The second way to calculate training auc
  rf_p_train <- as.vector(rf.model$votes[,2])
  rf_pr_train <- prediction(rf_p_train, train.data$class);
  r_auc_train2 <- performance(rf_pr_train, measure = "auc")@y.values[[1]]
  #cat("AUC =",r_auc_train2,"\n")
  
  # Quickly check equivalancy
  #all.equal(r_auc_train1,r_auc_train2)
  

  # Test AUC
  # The first way to calculate training auc
  rf_p_test <- predict(rf.model, type="prob",newdata = test.data)[,2]
  rf_pr_test <- prediction(rf_p_test, test.data$class)
  r_auc_test1 <- performance(rf_pr_test, measure = "auc")@y.values[[1]] 
  #cat("AUC =",r_auc_test,"\n")
  
  
  # Populate test and train auc
  rf_train_auc[m] <- r_auc_train1
  rf_test_auc[m]  <- r_auc_test1
  
  
}
# Quick check on simple boxplots 
boxplot(rf_train_auc)
boxplot(rf_test_auc)
# Total Random Forest time
sum(rf.time)

# Create dataframes to store values for plots
rftr = data.frame(Model = "Random Forest", value = rf_train_auc )

rft = data.frame(Model = "Random Forest", value = rf_test_auc )

#plot.test = rbind(rtt,ent,lat,rft) # this function will bind or join the rows.
#plot.train = rbind(rtr,entr,latr,rftr)

# Train and Test AUC Boxplots
box.test <-ggplot(rft, aes(x=Model, y=value, fill=Model)) + 
  geom_boxplot()+    
  labs(title = "Boxplots of Model",
       subtitle = "based on Test AUC")

box.train <-ggplot(rftr, aes(x=Model, y=value, fill=Model)) +  
  geom_boxplot() + labs(title = "Boxplots of Models",
                        subtitle = "based on Training AUC")
grid.arrange(box.train, box.test, nrow = 1) # Arrange Boxplots 



