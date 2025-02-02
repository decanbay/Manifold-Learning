clear all
close all
clc
pause(1)

%how many neighbors to consider when we create adjacency graph.
k=6; 
%how many corresponding pairs to use= p*N 
p=0.1; 

%sample rate.
step=floor(1/p);

%Retrieve data
%[X1, X2, X3]=getData;
X1 =csvread('x1.csv');
X1= X1(2:end,:);
X1= X1(:,2:end)';

X2 =csvread('x2.csv');
X2= X2(2:end,:);
X2= X2(:,2:end)';

fids=py.open(filename,'rb');
dataS=py.pickle.load(fids);

fidt=py.open(filename,'rb');
dataT=py.pickle.load(fidt);

tic
N=size(X1,2); %total number of instances.
%%
%Create kNN weight matrices.
idx1=knnsearch(X1', [], k);
idx2=knnsearch(X2', [], k);
W1=createKnnGraph(idx1);
W2=createKnnGraph(idx2);

%     %Create All connected graphs
%     delta=10;
%     W1=createAllConnectedGraph(X1,delta);
%     W2=createALLConnectedGraph(X2,delta);

%Create correspondence matrices
W12=sparse(N, N);
for i=1:step:N
    W12(i,i)=1;
end

%Make comparisons
epsilon=1e-8;
mu=1;


%Feature-level Manifold Projections
[map1, map2]=wmapGeneralTwo(X1, X2, W1, W2, W12, epsilon, mu);
visualize2(X1, X2, map1(:,1:3)'*X1, map2(:,1:3)'*X2, 4);


%Unsupervised alignment
%This one will take 3 minutes to run, so I comment it.
%The third parameter specifies how many neighbors to consider.
W12=generateWeight3(X1, X2, 8); 
[map1, map2]=wmapGeneralTwo(X1, X2, W1, W2, W12, epsilon, mu);
visualize2(X1, X2, map1(:,1:3)'*X1, map2(:,1:3)'*X2, 4);
toc