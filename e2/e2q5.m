% e2 q5

X = ( -6:0.2:6)';
Y = sinc(X) + 0.1.* rand( size(X));

out = [15 17 19];
Y( out) = 0.7 + 0.3* rand( size( out));
out = [41 44 46];
Y( out) = 1.5 + 0.2* rand( size( out));

model = initlssvm(X, Y, 'f', [], [], 'RBF kernel');
costFun = 'crossvalidatelssvm';
model = tunelssvm(model , 'simplex', costFun , {10 , 'mse'});

%[gam,sig2,cost] = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mse'}) ;

plotlssvm( model );


%% robust

model = initlssvm(X, Y, 'f', [], [], 'RBF kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm(model , 'simplex', costFun , {10 , 'mae';}, wFun );
model = robustlssvm( model );
plotlssvm( model );


%%

