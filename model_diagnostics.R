

#############################################################################################
####### TO RUN THIS SCRIPT YOU MUST HAVE RUN  PART OF THE R_CLASSIFICATION_CODE SCRIPT ######
#############################################################################################

elas.cv.fit      =     cv.glmnet(X.train, y.train, family = "binomial", alpha=0.5,nfolds=10)
elas.fit         =     glmnet(X.train, y.train, alpha = 0.5, lambda = elas.cv.fit$lambda.min)
end.time<-Sys.time()

beta0.elas.hat          =        elas.fit$a0
beta.elas.hat           =        as.vector(elas.fit$beta)
elas.prob.train         =        exp(X.train %*% beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X.train %*% beta.elas.hat +  beta0.elas.hat  ))


############ GRAPH SHOWING PREDICTED PROBABILITY OF DEFAULT VS ACTUAL STATUS#########
data$value<-exp(X%*% beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X %*% beta.elas.hat +  beta0.elas.hat  ))
ggplot(data%>%filter(class==0),aes(x=value,fill='Non-Bankrupt'))+geom_histogram(bins=75,alpha=0.5)+
  geom_histogram(data=data%>%filter(class==1),aes(x=value,fill='Bankrupt'),bins=75,alpha=0.5)+
  theme_minimal()+ggtitle('Training_data')+ xlab('predicted probability of default')+
  theme(
    legend.position=c(.2,.7),
    legend.justification = c(0, 0),
    legend.title = element_blank(),
    legend.background = element_rect(),
    legend.margin = margin(c(5, 5, 5, 0))
  )

############# GRAPH SHOWING PREDICTED PROBABILITY FOR COMPANIES THAT ACTUALLY WENT BANKRUPT#########
data$value<-exp(X%*% beta.elas.hat +  beta0.elas.hat  )/(1 + exp(X %*% beta.elas.hat +  beta0.elas.hat  ))
ggplot()+
  geom_histogram(data=data%>%filter(class==1),aes(x=value,fill='Bankrupt'),bins=75,alpha=0.5)+
  theme_minimal()+ggtitle('Training_data')+ xlab('predicted probability of default')

### GRAPH SHOWING ELASTIC NET MISCLASSIFICATION RATES FOR POSITIVES/NEGATIVES####
ggplot(df.elas)+geom_point(data=df.elas,size=1,aes(x=threshold,y=false_positive_rate,color='predicted bankruptcy \n but did not bankrupt'))+
  geom_point(data=df.elas,aes(x=threshold,y=false_negative_rate,color='predicted non-bankruptcy \n but went bankrupt'),size=1)+
  theme_minimal()+
  theme(
    legend.position=c(.05,.25),
    legend.justification = c(0, 0),
    legend.title = element_blank(),
    legend.background = element_rect(),
    legend.margin = margin(c(5, 5, 5, 0))
  )

########GRAPH SHOWING ELASTIC NET ERROR RATES######
ggplot(df.elas,aes(x=threshold,y=error_rate))+geom_line()+
  theme_minimal()
