
url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

using Knet

@doc download

data=readdlm(download(url))

y=data[:,14]
y=reshape(y,(1,506))

x=data[:,1:13]'
reshape(x,(13,506))

x_mean=mean(x,2)

x_std=std(x,2)

x_norm= (x.-x_mean)./x_std

srand(1)
A=randperm(506)
trn=A[1:400]
tst=A[401:506]

xtrn=x_norm[:,trn]

xtst=x_norm[:,tst]

ytrn=y[:,trn]

ytst=y[:,tst]

weight=randn(1,13)*0.1

predict(w,x)=w*x

ypred=predict(weight,xtrn)

loss(w,x,y)=sum((y-predict(w,x)).^2)/(2*length(y))

results=Dict("trainloss"=>loss(weight,xtrn,ytrn),"testloss"=>
loss(weight,xtst,ytst))

error=abs.(ytrn-ypred)


sum(error.<sqrt(results["trainloss"]))
println(sum(error))
