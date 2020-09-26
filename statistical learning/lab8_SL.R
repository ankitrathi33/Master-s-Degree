setwd("/Users/rathi/Downloads/Set13/Set13")
library(mclust)
library(umap)
library(dplyr)
library(ggplot2)
library(gplots)
library(dendextend)


# load files

df_1=read.csv("setA.txt", sep=' ')
df_2=read.csv("setB.txt", sep=' ')



data2<- data[,c("Gsto1", "Gstm1", "Cd9", "Prdx1", "Coro1a","Rac2", "Perp","Tissue")]
data3<- data[,c("Gsto1", "Gstm1", "Cd9", "Prdx1", "Coro1a","Rac2", "Perp")]



center_mean <- function(x) {
  ones = rep(1, nrow(x))
  x_mean = ones %*% t(colMeans(x))
  x - x_mean
}



center_data3<- center_mean(data3) 
cov <- cov(center_data3)


e <- eigen(cov)
e$values
# biggest eigenvalue 
Tissue<- data2$Tissue



my_pca<-function(df_1){
  
  m=as.matrix(scale(df_1)) #center scaling and making it a matrix
  m=cov(m)   #covariance matrix
  eigenv=eigen(m) 
  eigenv_1=-eigenv$vectors
  return(as.matrix(scale(df_1)) %*%eigenv_1)
}

features<-c('Gsto1', 'Gstm1', 'Cd9', 'Prdx1', 'Coro1a', 'Rac2', 'Perp')
pca_df<-my_pca(df_1[,features])

#scatter plot for 
color_map<-data.frame(
  v1=c("Bladder", "Marrow", "Colon", "Skin", "Spleen", "Tongue"),
  v2=c('red','blue','green','yellow','black','navy'),
  stringsAsFactors = F)

df_1_col<-df_1%>%left_join(color_map,
                           by=c('Tissue'='v1'))
plot(pca_df[,1],pca_df[,2],
     col=df_1_col[,'v2'])


#scree plot of percent variance explained

variance_explained<-pca_df%>%apply(2,sd)
perc_varianc_explained<-variance_explained/sum(variance_explained)
plot(1:length(perc_varianc_explained),
     100*cumsum(perc_varianc_explained),'l',
     xlab='component',
     ylab='variantion_explained',
     main='Percent Variance Explained')


#scree plot of percent variance explained

variance_explained<-pca_df%>%apply(2,sd)
perc_varianc_explained<-variance_explained/sum(variance_explained)
plot(1:length(perc_varianc_explained),
     100*cumsum(perc_varianc_explained),'l',
     xlab='component',
     ylab='variantion_explained',
     main='Percent Variance Explained')

#predict for dataset B


pca_1<-prcomp(df_1[,features],scale. = T)
prediction_for_set_b=predict(pca_1,df_2[,features])

df_2 <- as.numeric(as.character(unlist(df_2[[1]])))
df_2 <-data.matrix(df_2)
pca_1 <- prcomp(df_2, scale. = TRUE)
plot(pca_1)
biplot(pca_1)
summary(pca_1)


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


#screeplot

variance_explained_overall<-pca_overall$sdev/sum(pca_overall$sdev)
variance_explained_overall_cum<-cumsum(variance_explained_overall)

plot(1:20,
     100*variance_explained_overall_cum[1:20],'l',
     xlab='component',
     ylab='variantion_explained',
     main='Percent Variance Explained')

#Q8Perform UMAP for the whole dataset A. Compare two distance metrics: Euclidean and Pearson. For
# of them draw the scatterplot of the results with cells from various tissues marked differently.

#Euclidean
umap.defaults
config_eculidiean=umap.defaults

umpa_df_a = umap(df_1[,colnames(df_1)%>%setdiff('Tissue')],
                 config=config_eculidiean)


ggplot(umpa_df_a$layout%>%
         data.frame()%>%
         bind_cols(df_1%>%select("Tissue")),
       aes(x=X1,y=X2,color=Tissue))+geom_point()+ggplot2::ggtitle('Euclidean')



#Pearson
umap.defaults
config_pearson=umap.defaults
config_pearson$metric='pearson'

umpa_df_a_pearson = umap(df_1[,colnames(df_1)%>%setdiff('Tissue')],
                         config=config_pearson)


ggplot(umpa_df_a_pearson$layout%>%
         data.frame()%>%
         bind_cols(df_1%>%select("Tissue")),
       aes(x=X1,y=X2,color=Tissue))+geom_point()+ggplot2::ggtitle('Pearson')

#pearson displays better seggregation

umpa_df_b_pearson=predict(umpa_df_a_pearson,
                          df_2[,colnames(df_2)%>%setdiff('Tissue')])


ggplot(umpa_df_b_pearson%>%data.frame()%>%
         bind_cols(df_2%>%select("Tissue")),
       aes(x=X1,y=X2,color=Tissue))+geom_point()+ggplot2::ggtitle('Pearson on dataset B')

#clearly it can be seen that UMAP Pearson does a better job than PCA for dimensionality reduction



k.max <- 10
df_a_scaled <- df_1%>%select(-Tissue)%>%scale()
na_cols<-df_a_scaled%>%apply(2,sum_of_na)
non_na_cols<-na_cols[na_cols==0]%>%names()



wss_clusters<-c()
for(k in 2:k.max){
  print(k)
  wss<-kmeans(df_a_scaled[,non_na_cols], k, nstart=5,iter.max = 10 )$tot.withinss
  wss_clusters<-c(wss_clusters,wss)
  
}


plot(1:length(wss_clusters), wss_clusters,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")





pc1 <- as.matrix(data3) %*% e$vectors 
pc1_labeled<-cbind(pc1,Tissue) 
plot(pc1_labeled,col=c("red","blue","green","black","yellow","brown","silver")[Tissue])




data_unlabeled <- df_1 [,1:23433] 
data_labels <- data[,23434]

k.max <- 10
data <- data_unlabeled
wss <- vector("list", k.max-1)


for ( i in 1:k.max){
  k_means <- kmeans(data_unlabeled, i,nstart=5)
  
  wss[i] <- k_means$tot.withinss
}



plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")



k_means_optimal <- kmeans(data_unlabeled, 6,nstart=5)


labels_num <- as.numeric(factor(data_labels, levels = unique(data_labels)))
labels_num


table <- matrix(0,nrow=6, ncol=6,dimnames = list(c('Bladder', 'Colon', 'Marrow', 'Skin', 'Spleen',
                                                   'Tongue'), c(1:6)))


x <- matrix(table, dimnames = list(c("X","Y","Z","A","B","C"), c("A","B","C","X","Y","Z")))


for (i in 1:845){ 
  table[labels_num[i],k_means_optimal$cluster[i]] <- table[labels_num[i],k_means_optimal$cluster[i]]+1
}



data_umap <- umap(data_unlabeled, metric='pearson') 
Tissue<- data$Tissue
plot(data_umap$layout, col=c("red","blue","green","black","yellow","brown","silver")[Tissue]) Tissue<- data$Tissue
data_unlabeled <- subset(data, select = -c(Tissue))


pc<-prcomp(data_unlabeled)


pc_labeled<-cbind(pc$x,Tissue)
plot(pc_labeled,col=c("red","blue","green","black","yellow","brown","silver")[Tissue])





screeplot(pc, type='lines', npcs=20)


data_unlabeled <- df_1 [,1:23433]


data_umap <- umap(data_unlabeled, metric='pearson') Tissue<- $Tissue
plot(data_umap$layout, col=c("red","blue","green","black","yellow","brown","silver")[Tissue])


 
hc <- hclust(dist(df_2)) 
plot(hc)
plot(hc, hang = -1)


data_labels <- data[,23434]
data_unlabeled <- data [,1:23433] 
data_1st_1000 <- data[,1:1000]


new_data_mat <- as.matrix(data_1st_1000)
heatmap(new_data_mat, Colv=NA)


x <- c()


for( i in 1:1000){
  l<-length(unique(new_data_mat[,i]))
  if (l==1){
    x<-append(x,i)
  }
}


new_df<-data_1000[-x]
 
means<-colMeans(new_df) 
means_lev<-log10(means) 
View(means_lev)
num_of_bins=as.integer(sqrt(length(means_lev)))
hist(means_lev, breaks=num_of_bins)



our_gmm <- function(mu1,mu2,s1,s2, p1, p2) 
  {
  a=(s1^2-s2^2)/(2*s1^2*s2^2)
  b=(mu1*s2^2-mu2*s1^2)/(s1^2*s2^2)
  c=(-mu1^2*s2^2+mu2^2*s1^2)/(2*s1^2*s2^2)-log((s1*p2)/(s2*p1))
  
  
  delta=b^2-4*a*c
  
  x_1 = (-b+sqrt(delta))/(2*a) 
  x_2 = (-b-sqrt(delta))/(2*a) 
  result = c(x_1,x_2)
  
  return(result)
   }


z <- Mclust(means_lev)
our_booststrap <- MclustBootstrap(z)
plot(z)

