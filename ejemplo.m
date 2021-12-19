clear all
d =50;
h = 1/d;
Adiag = (2/h+(2*h)/3)*ones(d-2,1);
Anodiag = (h/6-1/h)*ones(d-3,1);
A  = diag(Anodiag,-1)+diag(Anodiag,1)+diag(Adiag,0);
f= (h/2) *ones(d-2,1);
sol = A\f;
y = [0;sol;0]
x = linspace(0,1,d);
plot(x,y)