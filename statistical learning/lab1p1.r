library(dplyr)
list.files('/Users/rathi/Downloads/Datasets')
df<-read.table('/Users/rathi/Downloads/Datasets/Datasets/part1_set12.txt',header = T)

#####################################Q1 
#Shapiro-Wilk test
shapiro.test(df$dose) 
shapiro.test(df$concentration) 

df$dose%>%hist(breaks=5)
df$concentration%>%hist(breaks=10)

#####################################Q2
##scatterplot
plot(df$concentration,df$dose)

#####################################Q3
#correlation
#pearson
pearson_corr <- cor.test(df$dose, df$concentration, 
                         method = "pearson")
pearson_corr  



#Spearman
spearman_corr <- cor.test(df$dose, df$concentration, 
                          method = "spearman")
spearman_corr  

#####################################Q4
#comment on scatterplot and correlation
#as it can be seen from the scatterplot and the correlation values the variables are negatively correlated to each other and can be written as a negative linear relationship between the two variables.

#####################################Q5
#linear model
model<-lm(df$concentration~df$dose)
model

#  f-test
ftest <- f.test(df$concentration~df$dose , data = df)
ftest

# t-test
ttest = t.test(df$concentration~df$dose)
tetst

#######q7######
#coefficient of determination 
summary(model_new)$r.squared
#coefficient of determination  of old model
summary(model)$r.squared 

#####q8####
#residual plot
resid(model) #List of residuals
plot(density(resid(model)))
qqnorm(resid(model)
qqline(resid(model))

#####q9######


