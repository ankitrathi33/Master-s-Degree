setwd("/Users/jauharim/Documents/Golu Projects/Set13")
library(dplyr)
library(ggplot2)
library(gplots)
library(umap)
install.packages('dendextend')
library(dendextend)

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
setb_predicted_pca<-predict(pca_overall,
                        df_2[,non_zero_sd_feats%>%setdiff('Tissue')])


ggplot(setb_predicted_pca%>%data.frame()%>%
         dplyr::bind_cols(df_2%>%select(Tissue)), aes(x=PC1, y=PC2, color=Tissue)) +
 geom_point()+ ggplot2::ggtitle('PCA results on set B')



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

#Q9 . Choose the metric that allows for better tissue separation
#going with pearson i.e. umpa_df_a_pearson


#.Q10 Perform the prediction on the dataset B. Draw the scatterplot to visually evaluate the results of

umpa_df_b_pearson=predict(umpa_df_a_pearson,
                          df_2[,colnames(df_2)%>%setdiff('Tissue')])


ggplot(umpa_df_b_pearson%>%data.frame()%>%
         bind_cols(df_2%>%select("Tissue")),
       aes(x=X1,y=X2,color=Tissue))+geom_point()+ggplot2::ggtitle('Pearson on dataset B')

#clearly it can be seen that UMAP Pearson does a better job than PCA for dimensionality reduction

######Part 2
###K mean clustering

#Elbow Method for finding the optimal number of clusters
# Compute and plot wss for k = 2 to k = 50
#nstart=5 menas that it has 5 random sets

k.max <- 50
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

#choosing 5 clusters

#Q2Perform k-means clustering for the optimal number of clusters. Investigate the composition of each

k=5 #optimal clusters

kmeans_optimal<-kmeans(df_a_scaled[,non_na_cols], k, nstart=5,iter.max = 20)

df_a_scaled_kmeans<-df_1
df_a_scaled_kmeans<-df_a_scaled_kmeans%>%bind_cols(data.frame(cluster=kmeans_optimal$cluster))


cluster_table<-df_a_scaled_kmeans%>%
  dplyr::group_by(Tissue,cluster)%>%
  dplyr::summarise(count=n())%>%
  tidyr::spread(key=cluster,value=count,fill=0)
cluster_table


#Q3 Draw the scatterplot of the UMAP results with cells from various k-means clusters differently. Comment
umap_set_a<-data.frame(umpa_df_a_pearson$layout)
umap_set_a$cluster=kmeans_optimal$cluster

ggplot(umap_set_a%>%bind_cols(df_1%>%select("Tissue")),
       aes(x=X1,y=X2,color=Tissue))+
  geom_point()+
  ggplot2::ggtitle('Pearson')+
  facet_grid(cluster~.)

#Comment on the results. Are cells from different tissues separated? Are clusters homo- or

#Hetrogenous


#part 3
#q Perform hierarchical clustering and draw the dendrogram. Comment on the results. 

df_b_scaled <- df_2%>%select(-Tissue)%>%scale()
na_cols<-df_b_scaled%>%apply(2,sum_of_na)
non_na_cols<-na_cols[na_cols==0]%>%names()


dist_mat <- dist(df_b_scaled[,non_na_cols], method = 'euclidean')
hclust_avg <- hclust(dist_mat, method = 'average')
plot(hclust_avg)

#Draw the heatmap of the first 1000 features with the dendrogram corresponding to cells (not genes).

dend_r <- df_b_scaled[,non_na_cols] %>% dist(method = "euclidean") %>% hclust() %>% as.dendrogram 

dend_c <- t(df_b_scaled[,non_na_cols]) %>% dist(method = "euclidean") %>% hclust() %>% as.dendrogram %>%
  color_branches(k=1000)

gplots::heatmap.2(as.matrix(df_b_scaled[,non_na_cols]), 
                  srtCol = 35,
                  Colv = dend_c,
                  trace="row", hline = NA, tracecol = "darkgrey",         
                  margins =c(6,3),      
                  key.xlab = "no / yes",
                  denscol = "grey",
                  density.info = "density",
)


=

