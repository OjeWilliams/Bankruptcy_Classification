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

#standardizing data for regularization
#X<-apply(X,FUN=scale,MARGIN=2)


#info for splitting into train/test in loop
n        =    nrow(X)
p        =    ncol(X)
n.train        =     floor(0.9*n)
n.test         =     n-n.train

M              =     50

#empty vectors to store time
elas.time<-c(rep(0,M)) 
rid.time<-c(rep(0,M))
las.time<-c(rep(0,M))
rf.time<-c(rep(0,M))

#Vectors to store error rates



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
  start.time<-Sys.time()
  elas.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0.5,nfolds=10)
  elas.fit         =     glmnet(X.train, y.train, alpha = 0.5, lambda = elas.cv.fit$lambda.min)
  end.time<-Sys.time()
  elas.time[m]<-elas.time+(end.time-start.time)
  
  #getting beta coefficients and probabilities
  beta0.elas.hat          =        elas.fit$a0
  beta.elas.hat           =        as.vector(elas.fit$beta)
  elas.prob.train         =        exp(X.train %*% beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X.train %*% beta.elas.hat +  beta0.elas.hat  ))
  
  #creating vectors to store data 
  tr.thrs                 =       c(0:10000)
  tr.TPR                  =       c(0:10000)
  tr.FPR                  =       c(0:10000)
  test.thrs               =       c(0:10000)
  test.TPR                =       c(0:10000)
  test.FPR                =       c(0:10000)
  
  for (i in 0:10000){
    if (i==0){
      thrs=0
    } else {
      thrs=i/10000
    }
    y.hat.train             =        ifelse(elas.prob.train > thrs, 1, 0) #table(y.hat.train, y.train)
    FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train                 =        sum(y.train==1) # total positives in the data
    N.train                 =        sum(y.train==0) # total negatives in the data
    FPR.train               =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train               =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train         =        FPR.train
    typeII.err.train        =        1 - TPR.train
    tr.thrs[i+1] <- thrs
    tr.TPR[i+1] <- TPR.train
    tr.FPR[i+1] <- FPR.train
    
  }
  
  #Calculating AUC
  train_auc=AUC(x=tr.FPR, y=tr.TPR)
  train.df=data.frame(true_positive_rate=tr.TPR,false_positive_rate=tr.FPR,threshhold=tr.thrs)
  tr.auc.caption<-paste('Training ROC\nAUC = ',as.character(train_auc),sep='')
  
  #PLOTTING ROC CURVE
  ggplot(train.df,aes(x=false_positive_rate,y=true_positive_rate))+geom_point(size=0.5,shape=0,color='#00BFC4')+
    geom_line(data=roc_train_df,aes(x=false_positive_rate,y=true_positive_rate,color=tr.auc.caption))+
    theme_minimal()+xlab('False positive rate')+ylab('True positive rate')+
    geom_abline(linetype='dashed',intercept=0,slope=1)+
    theme(
      legend.spacing.x = unit(.5, "char"), # adds spacing to the left too
      legend.position=c(.74,.05),
      legend.justification = c(0, 0),
      legend.title = element_blank(),
      legend.background = element_rect(),
      legend.margin = margin(c(5, 5, 5, 0))
    )


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