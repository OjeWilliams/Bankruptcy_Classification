library(tidyverse)
library(glmnet)
library(DescTools)
set.seed(1453)

#downloading data directly from github repository 
data <- read_csv('https://raw.githubusercontent.com/Jorgelopez1992/bankruptcy_classification/master/bankruptcy_Train.csv',
               col_names = T)

#splitting data intro X matrix and Y vector
X <- as.matrix(data[,c(1:64)])
y <- as.matrix(data[,65])


#info for splitting into train/test in loop
n        =    nrow(X)
p        =    ncol(X)
n.train        =     floor(0.9*n)
n.test         =     n-n.train

M              =     50   #total loops 
num_thrs       =     100  #Number of threshholds for classification for logistic regression (0.01,0.02 etc.)


#empty vectors to store time values
elas.time<-c(rep(0,M)) 
rid.time<-c(rep(0,M))
las.time<-c(rep(0,M))
rf.time<-c(rep(0,M))

#Vectors to store error rates
elas.train.error <- c(rep(0,M))
elas.test.error <-c(rep(0,M))
las.train.error <- c(rep(0,M))
las.test.error <-c(rep(0,M))
rid.train.error <- c(rep(0,M))
rid.test.error <-c(rep(0,M))
rf.train.error <- c(rep(0,M))
rf.test.error <-c(rep(0,M))

for (m in c(1:M)) {
  #randomly sampling rows 
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  weight.vec <- ifelse(y.train==0, sum(y.train)/n.train, sum(y.train==0)/n.train) #WEIGHTS BASED ON FREQUENCY
  
  ######################################################################################  
  ###############  ELASTIC NET   #######################################################
  ###################################################################################### 
  
  start.time <- Sys.time()
  elas.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0.5,nfolds=10, weights=weight.vec)
  elas.fit         =     glmnet(X.train, y.train, alpha = 0.5, lambda = elas.cv.fit$lambda.min, weights=weight.vec)
  end.time<-Sys.time()
  elas.time[m]<-elas.time[m]+(end.time-start.time)
  
  #getting beta coefficients and probabilities
  beta0.elas.hat          =        elas.fit$a0
  beta.elas.hat           =        as.vector(elas.fit$beta)
  elas.prob.train         =        exp(X.train %*% beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X.train %*% beta.elas.hat +  beta0.elas.hat  ))
  
  #creating vectors to store loop data for each threshold 
  tr.thrs                 =       c(0:num_thrs)
  tr.TPR                  =       c(0:num_thrs)
  tr.FPR                  =       c(0:num_thrs)
  tr.type1error           =       c(0:num_thrs)
  tr.type2error           =       c(0:num_thrs)
  tr.error                =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.train             =        ifelse(elas.prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.train                =        sum(y.train[y.hat.train==0] == 1) #false negatives in the data
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    tr.FNR[i+1] <- FN.train/P.train
    tr.FP[i+1] <- FP.train
    tr.FN[i+1] <- FN.train
    tr.error[i+1] <- (FP.train+FN.train)/n.train
    tr.type1error[i+1]         =       typeI.err.train
    tr.type2error[i+1]         =       typeII.err.train
    
  }
  
  #dataframe of all thresholds and errors
  df.elas=data.frame(threshold=tr.thrs,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                     false_positive=tr.FP,false_negative=tr.FN,
                     type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error
                     )
  
  #finding the probability threshold where type1error & type2error  are as close as possible
  min.dif.index <- which.min(abs(df.elas$type1error-df.elas$type2error)) #saving it into index so it can be saved automatically
  min.dif.thrs <- df.elas[min.dif.index,1]
  elas.train.error[m] <- df.elas[min.dif.index,8]
  
  ##################################        Elastic net test set           ##############################################
  prob.test               =        exp(X.test %*%beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X.test %*% beta.elas.hat +  beta0.elas.hat   ))
  y.hat.test              =        ifelse(prob.test > min.dif.thrs,1,0) #table(y.hat.test, y.test)  
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  FN.test                 =        sum(y.test[y.hat.test==0] == 1) #false negatives in the test_data
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
  FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
  typeI.err.test          =        FPR.test
  typeII.err.test         =        1 - TPR.test
  
  #saving test error rate
  elas.test.error[m]<- (FN.test+FP.test)/n.test
  
  #reporting results of test data
  df.elas[nrow(df.elas)+1,]<-c(min.dif.thrs,
                  FPR.test,
                  FN.test/P.test,
                  FP.test,
                  FN.test,
                  typeI.err.test,
                  typeII.err.test,
                  (FN.test+FP.test)/n.test)
  
  cat('loop ',as.character(m),'\n')
  cat('elastic net test results','\n')
  print(df.elas[nrow(df.elas),])
  
  
  ######################################################################################  
  ###############        RIDGE   #######################################################
  ###################################################################################### 
  start.time<-Sys.time()
  
  rid.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0,nfolds=10,weights=weight.vec)
  rid.fit         =     glmnet(X.train, y.train, alpha = 0, lambda = rid.cv.fit$lambda.min,weights=weight.vec)
  
  end.time<-Sys.time()
  rid.time[m]<-rid.time[m]+(end.time-start.time)
  
  #getting beta coefficients and probabilities
  beta0.rid.hat          =        rid.fit$a0
  beta.rid.hat           =        as.vector(rid.fit$beta)
  rid.prob.train         =        exp(X.train %*% beta.rid.hat +  beta0.rid.hat  )/(1 + exp(X.train %*% beta.rid.hat +  beta0.rid.hat  ))
  
  #creating vectors to store loop data for each threshold 
  tr.thrs                 =       c(0:num_thrs)
  tr.TPR                  =       c(0:num_thrs)
  tr.FPR                  =       c(0:num_thrs)
  tr.type1error           =       c(0:num_thrs)
  tr.type2error           =       c(0:num_thrs)
  tr.error                =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.train             =        ifelse(rid.prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.train                =        sum(y.train[y.hat.train==0] == 1) #false negatives in the data
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    tr.FNR[i+1] <- FN.train/P.train
    tr.FP[i+1] <- FP.train
    tr.FN[i+1] <- FN.train
    tr.error[i+1] <- (FP.train+FN.train)/n.train
    tr.type1error[i+1]         =       typeI.err.train
    tr.type2error[i+1]         =       typeII.err.train
    
  }
  
  #dataframe of all thresholds and errors
  df.rid=data.frame(threshold=tr.thrs,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                    false_positive=tr.FP,false_negative=tr.FN,
                    type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error
  )
  
  #finding the probability threshold where type1error & type2error  are as close as possible
  min.dif.index <- which.min(abs(df.rid$type1error-df.rid$type2error)) #saving it into index so it can be saved automatically
  min.dif.thrs <- df.rid[min.dif.index,1]
  rid.train.error[m] <- df.rid[min.dif.index,8]
  
  ##################################        ridge net test set           ##############################################
  prob.test               =        exp(X.test %*%beta.rid.hat +  beta0.rid.hat  )/(1 + exp(X.test %*% beta.rid.hat +  beta0.rid.hat   ))
  y.hat.test              =        ifelse(prob.test > min.dif.thrs,1,0) #table(y.hat.test, y.test)  
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  FN.test                 =        sum(y.test[y.hat.test==0] == 1) #false negatives in the test_data
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
  FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
  typeI.err.test          =        FPR.test
  typeII.err.test         =        1 - TPR.test
  
  #saving test error rate
  rid.test.error[m]<- (FN.test+FP.test)/n.test
  
  #reporting results of test data
  df.rid[nrow(df.rid)+1,]<-c(min.dif.thrs,
                             FPR.test,
                             FN.test/P.test,
                             FP.test,
                             FN.test,
                             typeI.err.test,
                             typeII.err.test,
                             (FN.test+FP.test)/n.test)
  
  cat('ridge net test results','\n')
  print(df.rid[nrow(df.rid),])
  
  
  ######################################################################################  
  ###############        lasso   #######################################################
  ###################################################################################### 
  start.time<-Sys.time()
  
  las.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=1,nfolds=10,weights=weight.vec)
  las.fit         =     glmnet(X.train, y.train, alpha = 1, lambda = las.cv.fit$lambda.min,weights=weight.vec)
  
  end.time<-Sys.time()
  las.time[m]<-las.time[m]+(end.time-start.time)
  
  #getting beta coefficients and probabilities
  beta0.las.hat          =        las.fit$a0
  beta.las.hat           =        as.vector(las.fit$beta)
  las.prob.train         =        exp(X.train %*% beta.las.hat +  beta0.las.hat  )/(1 + exp(X.train %*% beta.las.hat +  beta0.las.hat  ))
  
  #creating vectors to store loop data for each threshold 
  tr.thrs                 =       c(0:num_thrs)
  tr.TPR                  =       c(0:num_thrs)
  tr.FPR                  =       c(0:num_thrs)
  tr.type1error           =       c(0:num_thrs)
  tr.type2error           =       c(0:num_thrs)
  tr.error                =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  min.dif.thrs            =       NULL
  
  #loop to find the probability threshold where type1error==type2error (or as close as possible)
  for (i in 0:num_thrs){
    if (i==0){
      thrs=0
    } else {
      thrs=i/num_thrs
    }
    y.hat.train             =        ifelse(las.prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    FN.train                =        sum(y.train[y.hat.train==0] == 1) #false negatives in the data
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    tr.FNR[i+1] <- FN.train/P.train
    tr.FP[i+1] <- FP.train
    tr.FN[i+1] <- FN.train
    tr.error[i+1] <- (FP.train+FN.train)/n.train
    tr.type1error[i+1]         =       typeI.err.train
    tr.type2error[i+1]         =       typeII.err.train
    
  }
  
  #dataframe of all thresholds and errors
  df.las=data.frame(threshold=tr.thrs,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                    false_positive=tr.FP,false_negative=tr.FN,
                    type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error
  )
  
  #finding the probability threshold where type1error & type2error  are as close as possible
  min.dif.index <- which.min(abs(df.las$type1error-df.las$type2error)) #saving it into index so it can be saved automatically
  min.dif.thrs <- df.las[min.dif.index,1]
  las.train.error[m] <- df.las[min.dif.index,8]
  
  ##################################        lasso net test set           ##############################################
  prob.test               =        exp(X.test %*%beta.las.hat +  beta0.las.hat  )/(1 + exp(X.test %*% beta.las.hat +  beta0.las.hat   ))
  y.hat.test              =        ifelse(prob.test > min.dif.thrs,1,0) #table(y.hat.test, y.test)  
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  FN.test                 =        sum(y.test[y.hat.test==0] == 1) #false negatives in the test_data
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  TN.test                 =        sum(y.hat.test[y.test==0] == 0)# negatives in the data that were predicted as negatives
  FPR.test                =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test                =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity = recall
  typeI.err.test          =        FPR.test
  typeII.err.test         =        1 - TPR.test
  
  #saving test error rate
  las.test.error[m]<- (FN.test+FP.test)/n.test
  
  #reporting results of test data
  df.las[nrow(df.las)+1,]<-c(min.dif.thrs,
                             FPR.test,
                             FN.test/P.test,
                             FP.test,
                             FN.test,
                             typeI.err.test,
                             typeII.err.test,
                             (FN.test+FP.test)/n.test)
  
  cat('lasso net test results','\n')
  print(df.las[nrow(df.las),])

  
  
  ######################################################################################  
  ###########################   random forest   ########################################
  ###################################################################################### 
  
  
  
  
}


