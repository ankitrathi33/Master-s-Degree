library(dplyr)
list.files('/Users/rathi/Downloads/Datasets')
df<-read.table('/Users/rathi/Downloads/Datasets/Datasets/part2_set12.txt',header = T)


#####################################Q1 
#Shapiro-Wilk test
shapiro.test(df$density)  #it is not a normal dist as p<0.005
shapiro.test(df$concentration) #it is not a normal dist as p<0.005

df$density%>%hist(breaks=5)
df$concentration%>%hist(breaks=10)

#####################################Q2
##scatterplot
plot(df$concentration,df$density)

#####################################Q3
#correlation
#pearson
pearson_corr <- cor.test(df$density, df$concentration, 
                         method = "pearson")
pearson_corr  #because the output of p value is 7.737e-06 <0.05 we can say that correlation exist 
#correlation value is -0.5861335 



#Spearman
spearman_corr <- cor.test(df$density, df$concentration, 
                          method = "spearman")
spearman_corr  #because the output of p value is 1.45e-12<0.05 we can say that correlation exist 
#correlation value is 0.8071393


#####################################Q4
#comment on scatterplot and correlation
#as it can be seen from the scatterplot and the correlation values the variables are negatively correlated to each other and can be writen as a negative linear relationship between the two variables


#####################################Q5
#linear model
model<-lm(df$concentration~df$density)
model

#lm(formula = df$density ~ df$concentration)

#Coefficients:
#  (Intercept)   df$density  
#28.87       -78.47  

#coeff =-78.47  

#residual plot

resid(model) #List of residuals
plot(density(resid(model))) #A density plot
qqnorm(resid(model)) # A quantile normal plot - good for checking normality
qqline(resid(model))



#####################################Q6
#box cox on density 
y<-df$concentration
x<-df$density


library(forecast)
# to find optimal lambda
lambda = BoxCox.lambda( x)
# now to transform vector
transformed_x = BoxCox( x, lambda)
#scatteerplot
plot(y,transformed_x)


#####################################Q7
#correlation
#pearson
pearson_corr <- cor.test(y, transformed_x, 
                         method = "pearson")
pearson_corr  #because the output of p value is 6.695e-15 <0.05 we can say that correlation exist 
#correlation value is -0.849047 


#Spearman
spearman_corr <- cor.test(y, transformed_x,
                          method = "spearman")
spearman_corr  #because the output of p value is 1.45e-12 <0.05 we can say that correlation exist 
#correlation value is -0.8071393 



#####################################Q8
#fischer test is  useed to compare corr coeff
cor.diff.test = function(x1, x2, y1, y2, method="pearson") {
  cor1 = cor.test(x1, x2, method=method)
  cor2 = cor.test(y1, y2, method=method)
  
  r1 = cor1$estimate
  r2 = cor2$estimate
  n1 = sum(complete.cases(x1, x2))
  n2 = sum(complete.cases(y1, y2))
  fisher = ((0.5*log((1+r1)/(1-r1)))-(0.5*log((1+r2)/(1-r2))))/((1/(n1-3))+(1/(n2-3)))^0.5
  
  p.value = (2*(1-pnorm(abs(fisher))))
  
  result= list(
    "cor1" = list(
      "estimate" = as.numeric(cor1$estimate),
      "p.value" = cor1$p.value,
      "n" = n1
    ),
    "cor2" = list(
      "estimate" = as.numeric(cor2$estimate),
      "p.value" = cor2$p.value,
      "n" = n2
    ),
    "p.value.twosided" = as.numeric(p.value),
    "p.value.onesided" = as.numeric(p.value) / 2
  )
  cat(paste(sep="",
            "cor1: r=", format(result$cor1$estimate, digits=3), ", p=", format(result$cor1$p.value, digits=3), ", n=", result$cor1$n, "\n",
            "cor2: r=", format(result$cor2$estimate, digits=3), ", p=", format(result$cor2$p.value, digits=3), ", n=", result$cor2$n, "\n",
            "diffence: p(one-sided)=", format(result$p.value.onesided, digits=3), ", p(two-sided)=", format(result$p.value.twosided, digits=3), "\n"
  ))
  return(result);
}

cor.diff.test(x,transformed_x,y,y,method="pearson")  #no sign difference 
cor.diff.test(x,transformed_x,y,y,method = 'spearman')   #no sign difference


#####################################Q9
#linear model after transformation
model_new<-lm(y~transformed_x)
model_new

#Call:
#  lm(formula = y ~ transformed_x)

#Coefficients:
#  (Intercept)  transformed_x  
#18.068         -1.542  
summary(model_new)
summary(model)

#both p value and f value suggest the model have become better


#####################################Q11
#residual plot of the new model 
resid(model_new) #List of residuals
plot(density(resid(model_new))) #A density plot
qqnorm(resid(model_new)) #A quantile normal plot - good for checking normality
qqline(resid(model_new))


#####################################Q12
#coefficient of determination 
summary(model_new)$r.squared 

#coefficient of determination  of old model
summary(model)$r.squared 

#####################################Q13
#model performance has become better after the transfromation as can be seen form Q12