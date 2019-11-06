n = 10000;
k = 100;
fun = @(t,alpha,p)(normpdf(t,alpha.*k.*p,alpha.*k.*p.*(1-p)).*((1-normcdf(norminv(((n-k)./n),k.*p,k.*p.*(1-p))-t,(1-alpha).*k.*p,(1-alpha).*k.*p.*(1-p))).^2));
ke = @(alpha,p)(integral(@(t)fun(t,alpha,p),-inf,norminv(((n-k)./n),k.*p,k.*p.*(1-p)))+(1-normcdf(norminv(((n-k)./n),k.*p,k.*p.*(1-p)),alpha.*k.*p,alpha.*k.*p.*(1-p))));

% fplot(@(alpha) ke(alpha,0.01),[0,1])
ke01 = @(alpha) (ke(alpha,0.001));
ke05 = @(alpha) (ke(alpha,0.9));

fplot(@(x)[ke01(x) ke05(x)],[0,1])

funfixed =  @(t,alpha,p,thres)(normpdf(t,alpha.*k.*p,alpha.*k.*p.*(1-p)).*((1-normcdf(thres-t,(1-alpha).*k.*p,(1-alpha).*k.*p.*(1-p))).^2));%kefixed = @(alpha,p,thres)(integral(@(t)funfixed(t,alpha,p,thres),0,thres)+(1-normcdf(thres,alpha.*k.*p,alpha.*k.*p.*(1-p))));
kef01 = @(alpha) (kefixed(alpha,0.01,3));
kef02 = @(alpha) (kefixed(alpha,0.01,2.5));
fplot(@(x)[kef01(x),kef02(x)],[0,1])
