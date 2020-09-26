setwd('/Users/jauharim/Documents/Golu Projects/lab_6/')


library(ggplot2)
library(stats4)
library(PMCMR)
library(ggVennDiagram)
library(Hmisc)
library(gplots)
library(qvalue)

proteindata<-read.csv('proteinsMayak.csv',stringsAsFactors = F)
status<-read.csv('status.csv',stringsAsFactors = F)


cor.test(status$Age,status$Dose.mGy,method='spearman')

ggplot(data=status,aes(x=Age,y=Dose.mGy))+geom_point()+scale_x_log10()+scale_y_log10()

library(stats4)
# transform the data to data matrix
proteinData <- data.matrix(proteindata,)
rownames(proteinData)<-proteindata[,1]
# perform linear regression modelling on one protein with dose
#for i in 1:nrow(proteinData)
modVec<-c()
for(i in 1:nrow(proteinData)){
mod_dose <- lm(proteinData[i,2:ncol(proteinData)] ~ status$Dose.mGy)
mod_age <- lm(proteinData[i,2:ncol(proteinData)] ~ status$Age)
value<-ifelse(BIC(mod_dose)<BIC(mod_age),'Dose',"Age")
modVec<-c(modVec,value)
}

table(modVec)

#Filtering
proteinData<-proteinData[modVec=='Dose',]
kruskalPvalue_vec<-c()
for(i in 1:nrow(proteinData)){
protein<-proteinData[i,2:ncol(proteinData)]
tmp <- cbind(Val=protein,Lab=status$Dose.group)
kruskalPvalue <- kruskal.test(Val ~ Lab, data=tmp)$p.value
kruskalPvalue_vec<-c(kruskalPvalue_vec,kruskalPvalue)
}


kruskalPadj <- p.adjust(kruskalPvalue_vec, method='BH')
#filtering sign out
table(kruskalPadj<0.01)

proteinData<-proteinData[kruskalPadj<0.01,]


library(PMCMR)

dunnettPvals <- matrix(numeric(0),dim(proteinData)[1],3)


for(i in 1:dim(proteinData)[1]){
  tmp <- data.frame(Val=proteinData[i,2:ncol(proteinData)],Lab=status$Dose.group)
  dunnettPvals[i,]<-c(dunn.test.control(tmp$Val,tmp$Lab)$p.value)
}
colnames(dunnettPvals) <- c("high","low","medium")
dunnettPvals <- data.frame(dunnettPvals)

library(ggVennDiagram)
x <- list(high=rownames(proteinData)[dunnettPvals$high<0.01],
          low=rownames(proteinData)[dunnettPvals$low<0.01],
          medium=rownames(proteinData)[dunnettPvals$medium<0.01])

ggVennDiagram(x)




library(Hmisc)
library(gplots)
# create sample labels with dose group name and dose in mGy
clusLab <- paste(status$Dose.group,status$Dose.mGy,"mGy",sep="_")
proteinData<-proteinData[,2:ncol(proteinData)]
colnames(proteinData) <- clusLab
# perform hierarchical clustering and assign result to variable
clusRes <- varclus(proteinData, similarity="spearman",
                   type="data.matrix", vname=clusLab)$hclust
# draw heatmap with sample dendrogram
heatmap.2(proteinData, col=redgreen(75), scale="row", key=FALSE,
          symkey=FALSE, density.info="none",
          trace="none", margins = c(11, 2),
          labRow = "",dendrogram="col")


write.csv(rownames(proteinData),"diffProt.csv",
          row.names = F, quote=F)
