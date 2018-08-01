library(glmnet)
library(MASS)
library(ROCR)

data<-read.csv("Huazhen_Joel_Ethan_Updated.csv")
CCR_find<-function(p){
  tab=table(y_test,testpredict>p)
  CCR=sum(diag(tab))/sum(tab)
  CCR
}

# remove extraneous variables
df<-within(data,rm(dm_ad,dm_lp,datelp6,X.1,X,ordtyr,ordlyr,datead6))

# create new interaction variables for consistency
df$consistent1<-df$ordtyr2*df$ordlyr2
df$consistent2<-df$consistent1*df$ord2ago
df$consistent3<-df$consistent2*df$ord3ago

# binary variables for orders
df$ordtb<-ifelse(df$ordtyr2>0,1,0)
df$ordlb<-ifelse(df$ordlyr2>0,1,0)
df$ord2b<-ifelse(df$ord2ago>0,1,0)
df$ord3b<-ifelse(df$ord3ago>0,1,0)

# binary consistency variables for orders
df$cons1b<-df$ordtb*df$ordlb
df$cons2b<-df$cons1b*df$ord2b
df$cons3b<-df$cons2b*df$ord3b

# gzro is the response variable
df$gzro<-ifelse(df$targdol>0,1,0)

df$targdol<-NULL

# split into training and test
train<-df[which(df$train==1),]

train <- train[which(rownames(train) != 37839),]
train <- train[which(rownames(train) != 38283),]
train <- train[which(rownames(train) != 90895),]

N<-ncol(train)

x<-train[,-N]
y<-train[,N]

test<-df[which(df$train!=1),]
x_test<-test[,-N]
y_test<-test[,N]
x_test<-sapply(x_test,as.numeric)
x<-sapply(x,as.numeric)

# str(x)

# LASSO using class

# fit the model using LASSO regression (this takes a minute or two)
cvfit_class<-cv.glmnet(x,y,family="binomial",type.measure = "class")

# Check that majority classification is the best
testpredict<-predict(cvfit_class,newx=x_test,type="response")
p_star=seq(0,1,0.01)
CCR_seq=sapply(p_star,CCR_find)
plot(p_star,CCR_seq)
max(CCR_seq)
p_star[which.max(CCR_seq)]
plot(cvfit_class)

# most coefficients should be zero
coef(cvfit_class)
# summary(cvfit)

# make predictions on the test data, calculate correct classification rate (CCR), 2x2 table
pred_class<-predict(cvfit_class,newx=x_test,type="class")
tab_class<-table(pred_class,y_test)
CCR_class<-sum(diag(tab_class))/sum(tab_class)
CCR_class
tab_class

pred<-predict(cvfit_class,newx=x_test,type="response")
pred<-prediction(pred,y_test)
perf<-performance(pred,"tpr","fpr")
plot(perf)

perf<-performance(pred,"auc")
perf

# output probabilities (what's actually used with the multiple regression model)
probs<-predict(cvfit_class,newx=x_test,s="lambda.min",type="response")
colnames(probs)<-"logistic probs"

# LASSO using deviance

cvfit_dev = cv.glmnet(x, y, family = "binomial", type.measure = "deviance")

# Check that majority classification is the best
testpredict<-predict(cvfit_dev,newx=x_test,type="response")
p_star=seq(0,1,0.01)
CCR_seq=sapply(p_star,CCR_find)
plot(p_star,CCR_seq)
max(CCR_seq)
p_star[which.max(CCR_seq)]

plot(cvfit_dev)
coef(cvfit_dev)
print(cvfit_dev)

pred_dev<-predict(cvfit_dev,newx=x_test,type="class")
tab_dev<-table(pred_dev,y_test)
CCR_dev<-sum(diag(tab_dev))/sum(tab_dev)
CCR_dev
tab_dev

pred<-predict(cvfit_dev,newx=x_test,type="response")
pred<-prediction(pred,y_test)
perf<-performance(pred,"tpr","fpr")
plot(perf)

perf<-performance(pred,"auc")
perf

# LASSO using AUC

cvfit_auc = cv.glmnet(x, y, family = "binomial", type.measure = "auc")

testpredict<-predict(cvfit_auc,newx=x_test,type="response")
p_star=seq(0,1,0.01)
CCR_seq=sapply(p_star,CCR_find)
plot(p_star,CCR_seq)
max(CCR_seq)
p_star[which.max(CCR_seq)]

plot(cvfit_auc)
coef(cvfit_auc)
print(cvfit_auc)

pred_auc<-predict(cvfit_auc,newx=x_test,type="class")
tab_auc<-table(pred_auc,y_test)
CCR_auc<-sum(diag(tab_auc))/sum(tab_auc)
CCR_auc
tab_auc

pred<-predict(cvfit_auc,newx=x_test,type="response")
pred<-prediction(pred,y_test)
perf<-performance(pred,"tpr","fpr")
plot(perf)

perf<-performance(pred,"auc")
perf


# LASSO using MAE

cvfit_mae = cv.glmnet(x, y, family = "binomial", type.measure = "mae")

testpredict<-predict(cvfit_mae,newx=x_test,type="response")
p_star=seq(0,1,0.01)
CCR_seq=sapply(p_star,CCR_find)
plot(p_star,CCR_seq)
max(CCR_seq)
p_star[which.max(CCR_seq)]

plot(cvfit_mae)
coef(cvfit_mae)
print(cvfit_mae)

pred_mae<-predict(cvfit_mae,newx=x_test,type="class")
tab_mae<-table(pred_mae,y_test)
CCR_mae<-sum(diag(tab_mae))/sum(tab_mae)
CCR_mae
tab_mae

pred<-predict(cvfit_mae,newx=x_test,type="response")
pred<-prediction(pred,y_test)
perf<-performance(pred,"tpr","fpr")
plot(perf)

perf<-performance(pred,"auc")
perf
performance(pred,"tpr")
performance(pred,"spec")
performance(pred,"prec")
performance(pred,"rec")
performance(pred,"f")

# STEPWISE REGRESSION
# train$train<-NULL
# train$lpuryear2<-NULL
# train$lp6_bin<-NULL
# train$falord<-NULL
# train$recency_numeric<-NULL
# train$recency<-as.numeric(train$recency_factor)
# train$recency_factor<-NULL
# train$y<-train$gzro
# train$gzro<-NULL
# best.logistic <-
#   bestglm(Xy = train,
#           family = binomial,          # binomial family for logistic
#           IC = "AIC")
# 
# best.logistic$BestModels
# summary(best.logistic$BestModel)
# save work for use in model

# GLM on selected variables from Class
fit_class<-glm(gzro~ordhist+falord+recency_numeric+ordtb+cons1b+cons2b+cons3b,data=train,family=binomial)
summary(fit_class)
testpredict<-predict(fit_class,newdata=test)
tab=table(test$gzro,testpredict>0.5)
CCR<-sum(diag(tab))/sum(tab)
CCR

fit_class<-glm(gzro~falord+recency_numeric+cons1b,data=train,family=binomial)
summary(fit_class)

CCR_seq=sapply(p_star,CCR_find)
plot(p_star,CCR_seq)

p=p_star[which.max(CCR_seq)]

testpredict<-predict(fit_class,newdata=test,type="response")
tab=table(test$gzro,testpredict>p)
CCR<-sum(diag(tab))/sum(tab)
CCR
pred<-prediction(testpredict,y_test)
performance(pred,"auc")

testpredict<-data.frame(testpredict)
colnames(testpredict)<-"logistic probs"
write.csv(testpredict,"logistic_probs.csv")

# GLM on all variables

fit_all<-glm(gzro~.,data=train,family=binomial)
summary(fit_all)
testpredict<-predict(fit_all,newdata=test)
CCR_seq=sapply(p_star,CCR_find)
plot(p_star,CCR_seq)

p=p_star[which.max(CCR_seq)]
max(CCR_seq)
