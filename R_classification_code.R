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
num_thrs       =     500  #Number of threshholds for classification (0.01,0.02 etc.)


#empty vectors to store time values
elas.time<-c(rep(0,M)) 
rid.time<-c(rep(0,M))
las.time<-c(rep(0,M))
rf.time<-c(rep(0,M))

#Vectors to store error rates
elas.train.error.vector <- c(rep(0,M))
elas.test.error.vector <-c(rep(0,M))
las.train.error.vector <- c(rep(0,M))
las.test.error.vector <-c(rep(0,M))
rid.train.error.vector <- c(rep(0,M))
rid.test.error.vector <-c(rep(0,M))
rf.train.error.vector <- c(rep(0,M))
rf.test.error.vector <-c(rep(0,M))

for (m in c(1:M)) {
  #randomly sampling rows 
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  ######################################################################################  
  ###############  ELASTIC NET   #######################################################
  ###################################################################################### 
  
  #WEIGHTS
  weight.vec <- ifelse(y.train==0, sum(y.train)/n.train, sum(y.train==0)/n.train)
  
  start.time <- Sys.time()
  elas.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0.5,nfolds=10, weights=weight.vec)
  elas.fit         =     glmnet(X.train, y.train, alpha = 0.5, lambda = elas.cv.fit$lambda.min, weights=weight.vec)
  end.time<-Sys.time()
  elas.time[m]<-elas.time+(end.time-start.time)
  
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
  tr.FPR                  =       c(0:num_thrs)
  tr.FNR                  =       c(0:num_thrs)
  tr.FP                   =       c(0:num_thrs)
  tr.FN                   =       c(0:num_thrs)
  elas.train.error        =       NULL
  elas.test.error         =       NULL
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
    FNR.train               =        FN.train/N.train
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
  df.elas=data.frame(threshold=tr.thrs,true_positive_rate=tr.TPR,false_positive_rate=tr.FPR,false_negative_rate=tr.FNR,
                     type1error=tr.type1error,type2error=tr.type2error,error_rate=tr.error,
                     false_positive=tr.FP,false_negative=tr.FN)
  #finding the probability threshold where type1error & type2error  are as close as possible
  min.dif.index <- which.min(abs(df.elas$type1error-df.elas$type2error)) #saving it into index so it can be saved automatically
  min.dif.thrs <- df.elas[min.dif.index,1]
  elas.train.error <- df.elas[min.dif.index,-1]
  print(df.elas[min.dif.index,])
  view(df.elas)
  df.elas[which.min(abs(df.elas$error_rate)),]
  
  ######################################################################################  
  ###############        RIDGE   #######################################################
  ###################################################################################### 
  start.time<-Sys.time()
  rid.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0,nfolds=10)
  rid.fit         =     glmnet(X.train, y.train, alpha = 0, lambda = rid.cv.fit$lambda.min)
  
  end.time<-Sys.time()

  
  ######################################################################################  
  ###############        LASSO   #######################################################
  ###################################################################################### 
  start.time<-Sys.time()
  las.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0,nfolds=10)
  las.fit         =     glmnet(X.train, y.train, alpha = 0, lambda = las.cv.fit$lambda.min)
  end.time<-Sys.time()

  
#  cat( '\n',m,
#       '\nRsq.test.rf', Rsq.test.rf[m], 'Rsq.train.rf', Rsq.train.rf[m],
#       '\nRsq.test.elas', Rsq.test.elas[m],'Rsq.train.elas',Rsq.train.elas[m],
#       '\nRsq.test.rid', Rsq.test.rid[m],'Rsq.train.rid',Rsq.train.rid[m],
#       '\nRsq.test.las',Rsq.test.las[m],'Rsq.train.las',Rsq.train.las[m])
}
