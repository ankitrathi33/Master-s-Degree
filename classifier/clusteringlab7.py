%% GENERATE TEST DATASET (WITH THE SAME PARAMETRES AS IN D.C.I) AND LOAD THE YEAST DATASET ALSO IN D.C.II 
R1 = mvnrnd([4 7],[1 0; 0 1],100);
R2 = mvnrnd([1 1],[1 0; 0 1],100); R3 = mvnrnd([7 1],[1 0; 0 1],100); 
test_dataset_3 = [R1; R2; R3];
test_dataset_3_labels = [ones(1,100), ones(1,100)*2, ones(1,100)*3];

R1 = mvnrnd([6.5 6.5],[1 0; 0 1],150); R2 = mvnrnd([1 1],[1 0; 0 1],150); test_dataset_2 = [R1; R2];
test_dataset_2_labels = [ones(1,150), ones(1,150)*2];

test_dataset_1 = mvnrnd([5 5],[1 0; 0 1],300);

yeast_dataset = load('yeast.mat');
yeast_dataset = yeast_dataset.drozdze;

%% GENERATE MUTLICLASS SYNTHETIC DATASET
d1 = randn(50,100);
d2 = randn(50,100)+3;
d3 = randn(50,100); d3(1,:) = d3(1,:) + 9; d = [d1 d2 d3];
cLabel = [ones(1,100), ones(1,100)*2, ones(1,100)*3];

%% HIERARCHICAL CLUSTERING:
methods = {'average','complete','single'};
ways = {'euclidean','spearman'};
result = zeros(40,6);
k=1;

for i=1:length(methods)
for j=1:length(ways)
% Present in the form of histogram
gcp = clustergram(d,'Standardize', 'Row');
set(gcp,'Linkage',methods{i},'RowPdist',ways{j}); % the dendogram was setted to 3 as the number of clusters inside
Y = pdist(d', ways{j}); Y = squareform(Y);
Z = linkage(Y,methods{i});
T = cluster(Z,'cutoff',50,'Criterion','distance'); % cutoff criterion: at least 30 items in the cluster
% save the number of items and clusters in the table for z = 1:length(unique(T)) 

end
 
result(z,k) = length(find(T==z)); 


end
 

end
 
k = k+1; 

%% KOHONEN CLUSTERING:

% default parametres:
% no_neurons = 3
% topology = hextop
% epochs = 100
% layer = [1 3]
% Euclidean distance

% the proposed parametres checking during the excercise no_neurons = {[1 2], [1 3], [1 4], [1 5]};
topology = {'hextop','gridtop','randtop'};
epochs = [100, 200, 500, 1000];
layer = {[1 3], [1 2 3], [1 2 2 3], [1 2 2 2 3]};

% NUMBER OF NEURONS CHECKING
for i=1:length(no_neurons)
layer_first = no_neurons{i};
neuron_first = layer_first(length(layer_first)); net = selforgmap(no_neurons{i},100,3, 'hextop'); net.trainParam.epochs = 100;
net = train(net, d);

distances = dist(d', net.IW{1}'); [min_dist, cndx] = min(distances, [], 2);
 






end
 
figure()
scatter(d(1,:), d(2,:),200, cLabel, 'filled');
hold on
scatter(d(1,:), d(2,:),50, cndx, 'filled');
hold on
scatter(net.IW{1}(:,1), net.IW{1}(:,2),400, 'blackx')
title(sprintf('Number of neurons (or clusters): %d', neuron_first)) 

% LAYER SIZE CHECKING
for i=1:length(layer)
net = selforgmap(layer{i},100,3, 'hextop');
net.trainParam.epochs = 100;
net = train(net, d);

distances = dist(d', net.IW{1}'); [min_dist, cndx] = min(distances, [], 2);
 






end
 
figure()
scatter(d(1,:), d(2,:),200, cLabel, 'filled');
hold on
scatter(d(1,:), d(2,:),50, cndx, 'filled');
hold on
scatter(net.IW{1}(:,1), net.IW{1}(:,2),400, 'blackx')
title(sprintf('%i',layer{i})) 

% DISTANCE FUNCTION
net = selforgmap([1 3],100,3, 'hextop');
net.trainParam.epochs = 100;
net = train(net, d);

distances = dist(d', net.IW{1}'); [min_dist, cndx] = min(distances, [], 2);

figure()
scatter(d(1,:), d(2,:),200, cLabel, 'filled');
hold on
scatter(d(1,:), d(2,:),50, cndx, 'filled');
hold on
scatter(net.IW{1}(:,1), net.IW{1}(:,2),400, 'blackx')
title('Euclidean distance')
distancesbox = boxdist(d', net.IW{1}'); [min_distbox, cndxbox] = min(distancesbox, [], 2);

figure()
scatter(d(1,:), d(2,:),200, cLabel, 'filled');
hold on
scatter(d(1,:), d(2,:),50, cndxbox, 'filled');
hold on
scatter(net.IW{1}(:,1), net.IW{1}(:,2),400, 'blackx')
title('Box distance')

% TRAIN EPOCH NUMBERS
for i=1:length(epochs)
net = selforgmap([1 3],100,3, 'hextop');
net.trainParam.epochs = epochs(i);
net = train(net, d);

distances = dist(d', net.IW{1}'); 
[min_dist, cndx] = min(distances, [], 2);
 






end
 
figure()
scatter(d(1,:), d(2,:),200, cLabel, 'filled');
hold on
scatter(d(1,:), d(2,:),50, cndx, 'filled');
hold on
scatter(net.IW{1}(:,1), net.IW{1}(:,2),400, 'blackx')
title(sprintf('Number of epochs: %d',epochs(i))) 

% TOPOLOGY
for i=1:length(topology)
net = selforgmap([1 3],100,3, topology{i});
net.trainParam.epochs = 100;
net = train(net, d);

distances = dist(d', net.IW{1}'); [min_dist, cndx] = min(distances, [], 2);
 






end
 
figure()
scatter(d(1,:), d(2,:),200, cLabel, 'filled');
hold on
scatter(d(1,:), d(2,:),50, cndx, 'filled');
hold on
scatter(net.IW{1}(:,1), net.IW{1}(:,2),400, 'blackx')
title(sprintf('Topology: %s',topology{i})) 

%% PROVE THE CLUSTERITY FUNCTIONALITY FOR ARTIFICAL DATASET
% for 1-class dataset
net_1class = selforgmap([1 2 3],100, 3, 'hextop');
net_1class.trainParam.epochs = 100;
net_1class = train(net_1class, test_dataset_1');

test1_distances = dist(test_dataset_1, net_1class.IW{1}'); [min_dist, cndx] = min(test1_distances, [], 2);

figure()
scatter(test_dataset_1(:,1), test_dataset_1(:,2),50, cndx, 'filled');
hold on
scatter(net_1class.IW{1}(:,1), net_1class.IW{1}(:,2),200, 'blackx')
title('Kohonen clustering for 1-class artificial dataset')

gcp = clustergram(test_dataset_1','Standardize', 'Row');
set(gcp,'Linkage','average','RowPdist','euclidean'); % the dendogram was setted to 3 as the number of clusters inside

% for 2-class dataset
net_1class = selforgmap([1 3],100, 3, 'hextop');
net_1class.trainParam.epochs = 100;
net_1class = train(net_1class, test_dataset_2');

test1_distances = dist(test_dataset_2, net_1class.IW{1}'); [min_dist, cndx] = min(test1_distances, [], 2);

figure()
scatter(test_dataset_2(:,1), test_dataset_2(:,2),100, test_dataset_2_labels, 'filled');
hold on
scatter(test_dataset_2(:,1), test_dataset_2(:,2),50, cndx, 'filled');
hold on
scatter(net_1class.IW{1}(:,1), net_1class.IW{1}(:,2),200, 'blackx')
title('Kohonen clustering for 2-class artificial dataset')

gcp = clustergram(test_dataset_2','Standardize', 'Row');
set(gcp,'Linkage','average','RowPdist','euclidean'); % the dendogram was setted to 3 as the number of clusters inside

% for 3-class dataset
net_1class = selforgmap([1 3],100, 3, 'hextop');
net_1class.trainParam.epochs = 100;
net_1class = train(net_1class, test_dataset_3');

test1_distances = dist(test_dataset_3, net_1class.IW{1}'); [min_dist, cndx] = min(test1_distances, [], 2);

figure()
scatter(test_dataset_3(:,1), test_dataset_3(:,2),100, test_dataset_3_labels, 'filled');
hold on
scatter(test_dataset_3(:,1), test_dataset_3(:,2),50, cndx, 'filled');
hold on
scatter(net_1class.IW{1}(:,1), net_1class.IW{1}(:,2),200, 'blackx')
title('Kohonen clustering for 3-class artificial dataset')

gcp = clustergram(test_dataset_3','Standardize', 'Row');
set(gcp,'Linkage','average','RowPdist','euclidean'); % the dendogram was setted to 3 as the number of clusters inside


%% PROVE THE CLUSTERITY FUNCTIONALITY FOR YEAST DATASET net_yeast = selforgmap([1 2 3],100, 3, 'hextop'); net_yeast.trainParam.epochs = 100;
net_yeast = train(net_yeast, yeast_dataset'); 
yeast_distances = dist(yeast_dataset, net_yeast.IW{1}'); [min_dist, cndx] = min(yeast_distances, [], 2);

figure()
% 1 vs 1 procedure for Kohonen clustering subplot(2,3,1)
scatter(yeast_dataset(:,2), yeast_dataset(:,3),50, cndx, 'filled');
hold on
scatter(net_yeast.IW{1}(:,2), net_yeast.IW{1}(:,3),200, 'blackx')
title('2 vs 3')

subplot(2,3,2)
scatter(yeast_dataset(:,3), yeast_dataset(:,4),50, cndx, 'filled');
hold on
scatter(net_yeast.IW{1}(:,3), net_yeast.IW{1}(:,4),200, 'blackx')
title('3 vs 4')

subplot(2,3,3)
scatter(yeast_dataset(:,4), yeast_dataset(:,5),50, cndx, 'filled');
hold on
scatter(net_yeast.IW{1}(:,4), net_yeast.IW{1}(:,5),200, 'blackx')
title('4 vs 5')

subplot(2,3,4)
scatter(yeast_dataset(:,5), yeast_dataset(:,6),50, cndx, 'filled');
hold on
scatter(net_yeast.IW{1}(:,5), net_yeast.IW{1}(:,6),200, 'blackx')
title('5 vs 6')

subplot(2,3,5)
scatter(yeast_dataset(:,6), yeast_dataset(:,7),50, cndx, 'filled');
hold on
scatter(net_yeast.IW{1}(:,6), net_yeast.IW{1}(:,7),200, 'blackx')
title('6 vs 7')

% Hierarchical clustering
gcp = clustergram(yeast_dataset','Standardize', 'Row');
set(gcp,'Linkage','average','RowPdist','euclidean'); % the dendogram was setted to 3 as the number of clusters inside


