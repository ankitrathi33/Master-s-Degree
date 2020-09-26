setwd("/Users/jauharim/Documents/Golu Projects/Set13")
library(dplyr)

df_1=read.delim("setA.txt",stringsAsFactors = F,sep=' ')
df_2=read.delim("setB.txt",stringsAsFactors = F,sep=' ')


#Q1
#Perform PCA step by step without the use of ready function for the part of the dataset A, consisting
#only expression levels for the following genes: Gsto1, Gstm1, Cd9, Prdx1, Coro1a, Rac2, Perp.

my_pca<-function(df){
  
  m=as.matrix(scale(df)) #center scaling and making it a matrix
  m=cov(m)   #covariance matrix
  eigenv=eigen(m) 
  eigenv_1=-eigenv$vectors
  return(as.matrix(scale(df)) %*%eigenv_1)
}

features<-c('Gsto1', 'Gstm1', 'Cd9', 'Prdx1', 'Coro1a', 'Rac2', 'Perp')


pca_df<-my_pca(df_1[,features])

#Q2
#scatter plot for 
color_map<-data.frame(
  v1=c("Bladder", "Marrow", "Colon", "Skin", "Spleen", "Tongue"),
  v2=c('red','blue','green','yellow','black','navy'),
  stringsAsFactors = F)

df_1_col<-df_1%>%left_join(color_map,
                           by=c('Tissue'='v1'))

plot(pca_df[,1],pca_df[,2],
     col=df_1_col[,'v2'])




#Q3
#scree plot of percent variance explained

variance_explained<-pca_df%>%apply(2,sd)
perc_varianc_explained<-variance_explained/sum(variance_explained)
plot(1:length(perc_varianc_explained),
     100*cumsum(perc_varianc_explained),'l',
     xlab='component',
     ylab='variantion_explained',
     main='Percent Variance Explained')
#Q4
#predict for dataset B


pca_1<-prcomp(df_1[,features],scale. = T)
prediction_for_set_b=predict(pca_1,df_2[,features])


#Q5
#with the use of ready function

sum_of_na<-function(x){
  return(sum(is.na(x)))
}

zero_sd<-apply(df_1,2,sd)

features_having_zero_sd<-zero_sd[zero_sd==0]

#removing the features that are constant i.e. have 0 sd
# colnames(df_1)%>%
#   setdiff(features_having_zero_sd%>%names())%>%length()

non_zero_sd_feats<-colnames(df_1)%>%
  setdiff(features_having_zero_sd%>%names())


numeric_feats<-df_1%>%apply(2,is.numeric)


df_1_a <- mutate_all(df_1, function(x) as.numeric(as.character(x)))

non_numeric_cols<-dplyr::select_if(df_1, is.numeric)%>%colnames()
non_numeric_cols<-colnames(df_1)%>%setdiff(non_numeric_cols)
non_numeric_cols # Tissue

pca_overall<-prcomp(df_1[,non_zero_sd_feats%>%setdiff('Tissue')],scale. = T)

#plotting top two pc1, pc2
color_map<-data.frame(
  v1=c("Bladder", "Marrow", "Colon", "Skin", "Spleen", "Tongue"),
  v2=c('red','blue','green','yellow','black','navy'),
  stringsAsFactors = F)

df_1_col<-df_1%>%left_join(color_map,
                           by=c('Tissue'='v1'))

plot(pca_df[,1],pca_df[,2],
     col=df_1_col[,'v2'])


plot(pca_overall$x[,c(1,2)],col=df_1_col[,'v2'])

#Q6
#percent variance explained
#screeplot

variance_explained_overall<-pca_overall$sdev/sum(pca_overall$sdev)
variance_explained_overall_cum<-cumsum(variance_explained_overall)

plot(1:20,
     100*variance_explained_overall_cum[1:20],'l',
     xlab='component',
     ylab='variantion_explained',
     main='Percent Variance Explained')


#Q7 prediction on setB
setb_predicted<-predict(pca_overall,
                        df_2[,non_zero_sd_feats%>%setdiff('Tissue')])

#q8 umap 

