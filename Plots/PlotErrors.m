m=csvread("crossvalpoly.csv");
plot(m(1,:),m(2:3,:),'-x',16,m(3,15),'ko')
legend('Train Performance','Validation Performance','Location','southeast')
title("Tuning the Polynomial Features")
grid minor
xlabel("Maximal Polynomial Degree")
ylabel("Performance")
%xlim([-16 6])
ylim([0.815 0.835])
