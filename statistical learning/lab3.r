list.files('/Users/rathi/Downloads/lab3dataset')
df_1<-read.csv('/Users/rathi/Downloads/lab3dataset/dataset11_part_I.txt',sep='')
df_2<-read.csv('/Users/rathi/Downloads/lab3dataset/dataset11_part_II.txt',sep='')
df_3<-read.csv('/Users/rathi/Downloads/lab3dataset/dataset11_part_III.txt',sep='')

library(MASS) 

#1.Single input

#contig table
tbl<-table(df_1$Status,df_1$no.of.fam.members)
chisq.test(tbl) 

#null hypothesis- reject/accept status is indpendent of number of family members
#alternate hypothesi- reject/accept  status is dependent of number of family members


#As the p-value 0.2477 is greater than the .05 significance level, 
#we do not reject the null hypothesis 
#X-squared = 4.1309, df = 3,



#contig table
tbl<-table(df_1$Status,df_1$no.of.fam.members)
chisq.test(tbl) 

#null hypothesis- reject/accept status is indpendent of number of family members
#alternate hypothesi- reject/accept  status is dependent of number of family members


#As the p-value 0.2477 is greater than the .05 significance level, 
#we do not reject the null hypothesis 
#X-squared = 4.1309, df = 3,


library(epitools)

## ratio CI

#RR ratio 
riskratio.wald(tbl)
# $measure measures the CI of Risk Ratio

#OR ratio
oddsratio.wald(tbl)
# $measure measures the CI of Odds Ratio


#logistic regresison model 
model <- glm (Status~no.of.fam.members,data=df_1, family = 'binomial')

#CI of beta value
confint(model, 'no.of.fam.members', level=0.95)
library(aod)

#wald test
aod::wald.test(Sigma = vcov(model),b = coef(model))

library(survey)

regTermTest(model, "Status")


#mcfdden

mod <- glm(Status~no.of.fam.members, data=df_1,family="binomial")
nullmod <- glm(Status~1, data=df_1,family="binomial")
mcfdden_rsquare<-1-(logLik(mod)/logLik(nullmod))

mcfdden_rsquare # it is close 0 

#LRT
anova(mod, nullmod, test = "LRT")

#final conclusions about model and accuracy 


#--------part2--------

#logistic regression model 

model <- glm (Status~Income,data=df_2, family = 'binomial')

#CI of beta value
confint(model, 'Income', level=0.95)
library(aod)

#wald test
aod::wald.test(Sigma = vcov(model),b = coef(model))

library(survey)

regTermTest(model, "Status")

#mcfdden

mod <- glm(Status~Income, data=df_2,family="binomial")
nullmod <- glm(Status~1, data=df_2,family="binomial")
mcfdden_rsquare<-1-(logLik(mod)/logLik(nullmod))

mcfdden_rsquare

#LRT
anova(mod, nullmod, test = "LRT")

#final conclusion about model and analysis 

#--------part3--------

library(dplyr)


#Train and Test #part 1
set.seed(10)
train_rows<-sample(1:nrow(df_3),nrow(df_3)*.8,replace=F)
test_rows<-setdiff(1:nrow(df_3),train_rows)
data_train<-df_3[train_rows,]
data_test<-df_3[test_rows,]


model <- glm (Status~.,data=data_train, family = 'binomial')

#BIC based feature selection #part 2
BIC<-stepAIC(model,k=log(nrow(data_train)))

BIC  #final model has Income and Credit History as the desired variables


#backward using BIC  #part 3

BIC_backward<-stepAIC(model,k=log(nrow(data_train)),direction = 'backward')

#backward using p value  #part 3
p_value_back<-step(model, direction = "backward", test = "F")

#LRT  #Part 4

#LRT

bic_coeff<-BIC_backward$coefficients
bic_coeff<-bic_coeff%>%data.frame()%>%row.names()%>%setdiff("(Intercept)")


mod_bic_back <- glm(Status~Income+Credit.history, data=data_train,family="binomial")
nullmod_bic_babck <- glm(Status~1, data=data_train,family="binomial")

anova(mod_bic_back, nullmod_bic_babck, test = "LRT")



p_value_coeff<-p_value_back$coefficients
p_value_coeff<-p_value_coeff%>%data.frame()%>%row.names()%>%setdiff("(Intercept)")


data_train%>%colnames()
data_train<-data_train%>%mutate(marriage_yes=ifelse(Marriage=='Yes',1,0),
                                Region.of.building_Countryside=ifelse(Region.of.building=='Countryside',1,0),
                                Region.of.building_Suburbs=ifelse(Region.of.building=='Suburbs',1,0))

data_test<-data_test%>%mutate(marriage_yes=ifelse(Marriage=='Yes',1,0),
                              Region.of.building_Countryside=ifelse(Region.of.building=='Countryside',1,0),
                              Region.of.building_Suburbs=ifelse(Region.of.building=='Suburbs',1,0))

mod_p_value_back <- glm(Status~marriage_yes+Income+Credit.in.thousend+Credit.history+Region.of.building_Countryside+Region.of.building_Suburbs, data=data_train,family="binomial")
null_p_value_back <- glm(Status~1, data=data_train,family="binomial")

anova(mod_p_value_back, null_p_value_back, test = "LRT")


#BIC value

BIC(mod_p_value_back)
BIC(mod_bic_back)



#plot roc auc 


library(ROCR)
pred1 <- prediction(predict(mod_bic_back), data_train$Status)
perf1 <- performance(pred1,"tpr","fpr")
plot(perf1)


library(pROC)
data(aSAH)
rocobj <- roc(data_train$Status, predict.glm(mod_p_value_back,data_train))
coords(rocobj, "best")
coords(rocobj, x="best", input="threshold", best.method="youden")

