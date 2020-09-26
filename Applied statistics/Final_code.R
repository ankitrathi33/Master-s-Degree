library(tidyverse)
library(magrittr)
library(plotly)
library(ggthemes)
library(knitr)
library(corrplot)
library(Hmisc)
library(PerformanceAnalytics)
library(mice)
library(randomForest)
library(caret)
library(ggplot2)
library(aod)
library(Amelia)
library(pscl)

african_data <- read.csv2('D:/Data Science/2nd Semester/Applied Statistics/AS Semester Laboratory/african_crises.csv', sep = ",", header = TRUE, na.strings=c(""))
african_data %>% head()
View(african_data)

sapply(african_data,function(x) sum(is.na(x))) ##We check if there are no NAs

sapply(african_data, function(x) length(unique(x))) ##Unique values

missmap(african_data, main = "Missing values vs observed")

#Model fitting

train <- african_data[1:848,]
test <- african_data[849:1059,]

model <- glm(banking_crisis ~.,family=binomial(link='logit'),data=train)
summary(model)

#Now we can run the anova() function on the model to analyze the table of deviance
anova(model, test="Chisq")

pR2(model)


