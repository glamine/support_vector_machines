% test roc

load iris_dataset

%%

net1 = patternnet(20);
net1 = train(net1,irisInputs,irisTargets);
irisOutputs = sim(net1,irisInputs);
[tpr,fpr,thresholds] = roc(irisTargets,irisOutputs)
plotroc(irisTargets,irisOutputs)

%%

load simplecluster_dataset

%%
net2 = patternnet(20);
net2 = train(net2,simpleclusterInputs,simpleclusterTargets);
simpleclusterOutputs = sim(net2,simpleclusterInputs);
[tpr,fpr,thresholds] = roc(simpleclusterTargets,simpleclusterOutputs)
plotroc(simpleclusterTargets,simpleclusterOutputs)