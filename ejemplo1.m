g1 = @(a,b) (3/2)*(b^2-a^2)-(2/3)*(b^3-a^3)
g2 = @(a,b) 3*(b-a)-(b^2-a^2)
u = @(x) x-x^2
x = linspace(0,1,d);
for i=2:(d-1)
	sum1 = -x(i-1)*g2(x(i-1),x(i))+g1(x(i-1),x(i));
	sum2 = x(i+1)*g2(x(i),x(i+1))-g1(x(i),x(i+1));
	f(i-1,1) =(1/h)*(sum1+sum2);
	uhat(i-1,1) = u(x(i));
end;
sol = A\f; y = [0;sol;0]; yy = [0;uhat;0];
plot(x,y); hold on; plot(x,yy);