setwd("/Users/jauharim/Documents/Golu Projects")
library(dplyr)
library(tidyr)
#install.packages('glmnet')
library(glmnet)
set.seed(10)
install.packages('psych')
library(psych) 

df=read.delim('dataset.txt',stringsAsFactors = F)

#one hot encoding
df<-df%>%
  mutate(one = 1) %>% 
  spread(Gender, one, fill = 0, sep = "") %>% 
  mutate(one = 1) %>% 
  spread(Married, one, fill = 0, sep = "") %>% 
  mutate(one = 1) %>% 
  spread(Ethnicity, one, fill = 0, sep = "")
#making the dependent variable by taking balance==0 as defaulters
df$dv=ifelse(df$Balance==0,1,0)
#split into train and test
train_rows<-sample(1:nrow(df),nrow(df)*0.8)
test_rows<-setdiff(1:nrow(df),train_rows)


df_train<-df[train_rows,]
df_test<-df[test_rows,]

#lambda values
lambda_values<-10^(seq(10,-2,-0.244))

features=colnames(df_train)%>%setdiff(c('Balance','dv'))
X_scaled <- scale(df_train[,features]%>%data.matrix())
y=df_train$dv

cv_fit<-cv.glmnet(x=X_scaled,
                  y=y,
                  nfold=10,
                  alpha = 0, lambda = lambda_values)

#plotting lambdas
plot(cv_fit)


#getting 1se lambda
best_lambda_1se<-cv_fit$lambda.1se

cv_fit$lambda.min


aic <- c()
bic <- c()
X_scaled <- scale(df_train[,features]%>%data.matrix())
y=df_train$dv
for (lambda in seq(lambda_values)) {
  # Run model
  model <- glmnet(x=X_scaled, 
                  y=y, alpha = 0, lambda = lambda_values[lambda], standardize = TRUE)
  # Extract coefficients and residuals (remove first row for the intercept)
  betas <- as.vector((as.matrix(coef(model))[-1, ]))
  resid <- y - (X_scaled %*% betas)
  # Compute hat-matrix and degrees of freedom
  ld <- lambda_values[lambda] * diag(ncol(X_scaled))
  H <- X_scaled %*% solve(t(X_scaled) %*% X_scaled + ld) %*% t(X_scaled)
  df <- psych::tr(H)
  # Compute information criteria
  aic[lambda] <- nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df
  bic[lambda] <- nrow(X_scaled) * log(t(resid) %*% resid) + 2 * df * log(nrow(X_scaled))
}

aic<-unlist(aic)
bic<-unlist(bic)

# Plot information criteria against tried values of lambdas
plot(log(lambda_values), bic, col = "orange", type = "l"
     , ylab = "Information Criterion")
lines(log(lambda_values), aic, col = "skyblue3")
legend("bottomright", lwd = 1, col = c("orange", "skyblue3"), legend = c("BIC", "AIC"))


# Optimal lambdas according to both criteria AIC  BIC
lambda_aic <- lambda_values[which.min(aic)]  #0.01
lambda_bic <- lambda_values[which.min(bic)] #0.1
lambda_mse<-best_lambda_1se #0.034


#models based on best aic bic and mse
model_aic <- glmnet(X_scaled, y, alpha = 0, lambda = lambda_aic, standardize = TRUE)
model_bic <- glmnet(X_scaled, y, alpha = 0, lambda = lambda_bic, standardize = TRUE)
model_lambda_mse <- glmnet(X_scaled, y, alpha = 0, lambda = lambda_mse, standardize = TRUE)

model_null = glm(y~1)

anova(model_null,model_aic, test = 'LRT')

anova(model_null,model_bic, test = 'LRT')

library(lmtest)
