clear all
d =50;
h = 1/d;
Adiag = (2/h+(2*h)/3)*ones(d-2,1);
Anodiag = (h/6-1/h)*ones(d-3,1);
A  = diag(Anodiag,-1)+diag(Anodiag,1)+diag(Adiag);
g1 = @(a,b) (3/2)*(b^2-a^2)-(2/3)*(b^3-a^3)
g2 = @(a,b) 3*(b-a)-(b^2-a^2)
u = @(x) x-x^2
x = linspace(0,1,d);
for i=2:d+1
	sum1 = -x(i-1)*g2(x(i-1),x(i))+g1(x(i-1),x(i));
	sum2 = x(i+1)*g2(x(i),x(i+1))-g1(x(i),x(i+1));
	f(i-1,1) =(sum1/(x(i)-x(i-1))+sum2/(x(i+1)-x(i)));
end;
for i=1:d 
	uhat(i,1) = u(x(i));
end;
sol = A\f; y = [0;sol;0]; 
plot(x,y); hold on; plot(x,uhat);