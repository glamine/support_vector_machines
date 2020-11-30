% e2 h2

load santafe ;

x = 1:1200;

figure(1) ;
plot(Z)

figure(2)
plot(Ztest) ;

figure(4)
plot(x(1:1000),Z,'b') ;
hold on;
plot(x(1000:1200),[Z(end,1);Ztest],'r')
legend('Train set','Test set')
hold off

%%

