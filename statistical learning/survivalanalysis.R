set.seed(185)  #mentioned in the seed


setwd('/Users/rathi/Downloads/survivAL ANALYSIS')

library(survival)
#install.packages('survminer')
library(survminer)
library(dplyr)

df_1<-read.csv('part1_censored_16.txt',sep='')
df_2<-read.csv('part1_uncensored_16.txt',sep='')

colnames(df_2)[2]<-"Uncensored"

df_3<-df_1%>%cbind(df_2)


surv_object <- Surv(time = df_3$Time, event = df_3$Censored)
surv_object 

#kaplan meier fit
KM0 <- survfit(Surv(Time, Censored) ~ 1,  type="kaplan-meier", conf.type="log", data=df_1)
KM1 <- survfit(Surv(Time, Uncensored) ~ 1,  type="kaplan-meier", conf.type="log", data=df_2)

#Q1
#table outputs  
summary(KM0)

summary(KM1)


#plotting the two 
#censored
plot(KM0, main=expression(paste("Kaplan-Meier-estimate ", hat(S)(t), " with CI for censored")),
     xlab="t", ylab="Survival", lwd=2)

plot(KM1, main=expression(paste("Kaplan-Meier-estimate ", hat(S)(t), " with CI for uncensored")),
     xlab="t", ylab="Survival", lwd=2)


#Or
plot(Surv(time = df_1$Time, event = df_1$Censored))
plot(Surv(time = df_2$Time, event = df_2$Uncensored))


#as can be seen from the data above censoring does effect the survivial time 


#Q2

df_4<-read.csv('part2_16.txt',sep='')



surv_object_1 <- Surv(time = df_4$Time, event = df_4$Censored)

fit_for_stage_factor <- survfit(surv_object ~ Stage, data = df_4)
summary(fit_for_stage_factor)

#plotting the log rank 

ggsurvplot(fit_for_stage_factor, data = df_4, pval = TRUE)

##Q1
#based on the above curve the stage factor doesn't influence survival time

#Q2
#based on the log rank test the value of p-value is 0.5 which is less than 0.05 i.e. p<0.05 and hence it is insignificant
#i.e. stage doesnt effect survival time 


#part 3


df_5<-read.csv('part3_16.txt',sep='')

surv_object_2 <- Surv(time = df_5$Time, event = df_4$Censored)

fit_for_therapy_factor <- survfit(surv_object_2 ~ Therapy, data = df_5)
summary(fit_for_therapy_factor)

#survivial curve of all three together
ggsurvplot(fit_for_therapy_factor, data = df_5, pval = TRUE)

#log rank test of surgery vs chemo

#converting time to numeric as it is in integer currently

df_5_a<-df_5%>%filter(Therapy %in% c('Chemotherapy','Surgery'))%>%mutate(time_1=as.numeric(Time))   
df_5_b<-df_5%>%filter(Therapy %in% c('Chemotherapy','Radiotherapy'))%>%mutate(time_1=as.numeric(Time))   
df_5_c<-df_5%>%filter(Therapy %in% c('Radiotherapy','Surgery'))%>%mutate(time_1=as.numeric(Time))  

#surg vs chemo 

surv_object_a <- Surv(time = df_5_a$Time, event = df_5_a$Censored)
fit_for_stage_factor_a <- survfit(surv_object_a ~ Therapy, data = df_5_a)
summary(fit_for_stage_factor_a)
ggsurvplot(fit_for_stage_factor_a, data = df_5_a, CI = TRUE)



#chemo vs radio 

surv_object_b <- Surv(time = df_5_b$Time, event = df_5_b$Censored)
fit_for_stage_factor_b <- survfit(surv_object_b ~ Therapy, data = df_5_b)
summary(fit_for_stage_factor_b)
ggsurvplot(fit_for_stage_factor_b, data = df_5_b, CI = TRUE)


#surg vs radio 

surv_object_c <- Surv(time = df_5_c$Time, event = df_5_c$Censored)
fit_for_stage_factor_c <- survfit(surv_object_c ~ Therapy, data = df_5_c)
summary(fit_for_stage_factor_c)
ggsurvplot(fit_for_stage_factor_c, data = df_5_c, CI = TRUE)


#plotting log rank for each

#surg vs chemo 

surv_object_a <- Surv(time = df_5_a$Time, event = df_5_a$Censored)
fit_for_stage_factor_a <- survfit(surv_object_a ~ Therapy, data = df_5_a)
summary(fit_for_stage_factor_a)
ggsurvplot(fit_for_stage_factor_a, data = df_5_a, pval = TRUE)



#chemo vs radio 

surv_object_b <- Surv(time = df_5_b$Time, event = df_5_b$Censored)
fit_for_stage_factor_b <- survfit(surv_object_b ~ Therapy, data = df_5_b)
summary(fit_for_stage_factor_b)
ggsurvplot(fit_for_stage_factor_b, data = df_5_b, pval = TRUE)


#surg vs radio 

surv_object_c <- Surv(time = df_5_c$Time, event = df_5_c$Censored)
fit_for_stage_factor_c <- survfit(surv_object_c ~ Therapy, data = df_5_c)
summary(fit_for_stage_factor_c)
ggsurvplot(fit_for_stage_factor_c, data = df_5_c, pval = TRUE)


#surgery does effect the survivial probabbility and time but the p_value is still >0.5 i.e. 0.056

##Part iv cox propotional 
df_6<-read.csv('part4.txt',sep='')
#type of columns 
#columns are in factor form so not converting yet into one hot encoding as mentioned in the assignment 



#splitting into train and test
set.seed(185)  #mentioned in the seed
train_rows<-sample(1:nrow(df_6),nrow(df_6)*.8,replace=F)
test_rows<-setdiff(1:nrow(df_6),train_rows)
df_train<-df_6[train_rows,]
df_test<-df_6[test_rows,]


#cox propotional model on train with the three features

res.cox <- coxph(Surv(Time, Censoring) ~ Age + Sex + Calories, data =  df_train)
summary(res.cox)  


#null model of cox propotional model
res.cox_null<-coxph(Surv(Time, Censoring) ~ 1, data =  df_train)
summary(res.cox_null)

#LR test to compare with the null model
anova(res.cox,res.cox_null,test="LRT")
#we reject the null hypothesis i.e. model doesn't depend on age+sex+calories as p value is 0.007245 <0.05

#wald test for the free variables
library(aod)
wald.test(b = coef(res.cox), Sigma = vcov(res.cox), Terms = 1) # wald test for age
wald.test(b = coef(res.cox), Sigma = vcov(res.cox), Terms = 2) # wald test for sex
wald.test(b = coef(res.cox), Sigma = vcov(res.cox), Terms = 3) # wald test for calories



#hazard raio 
exp(coef(res.cox))  #age ,sex=male and calories 

#ggforest plot
ggforest(res.cox)

# cox propotional hazard model
library(goftest)
res.cox <- coxph(Surv(Time, Censoring) ~ Age + Sex + Calories, data =  df_train)
summary(res.cox)
goodness.fit(AIC, BIC, starts, data = df_train, method = "PSO", domain = c(0,Inf),
             mle = NULL,...)

