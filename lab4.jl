using Knet
include(Pkg.dir("Knet","data","mnist.jl"))#include mnist related functions
xtrn,ytrn,xtst,ytst =mnist()#data ready to use in convolutional neural networks

function initweights(h)  # use cinit(x,h1,h2,...,hn,y) for n hidden layer model
    w = Any[]
    x = h[1]
    for i=2:length(h)
        if isa(h[i],Tuple)
            (x1,x2,cx) = x
            (w1,w2,cy) = h[i]
            push!(w, xavier(w1,w2,cx,cy))
            push!(w, zeros(1,1,cy,1))
            x = (div(x1-w1+1,2),div(x2-w2+1,2),cy) # with default padding and pooling parameters
        elseif isa(h[i],Integer)
            push!(w, xavier(h[i],prod(x)))
            push!(w, zeros(h[i],1))
            x = h[i]
        end
    end
    map(KnetArray{Float32},w)
end


function cnn(w,x)
    for i=1:2:length(w)
        if ndims(w[i])==4 #it means convolution layer
            x=conv4(w[i],x) .+ w[i+1]
            x=pool(sigm.(x))
        elseif ndims(w[i])==2 #it means fully connected layer
            x=w[i]*mat(x) .+ w[i+1]
            if i < length(w)-1
                x = sigm.(x)
            end
        end
    end
    return x
end

function loss(w,x,ygold)
    ypred=cnn(w,x)
    prob=logp(ypred,[1])
    J=-sum(ygold.*prob)
    return J
end

lossgradient=grad(loss)

function train(w, dtrn, lr)
        for (x,y) in dtrn
            g=lossgradient(w,x,y)
            update!(w,g,lr=lr)
        end
    return w

end

function accuracy(weights, dtst, pred=cnn)
    w = weights
    ncorrect = 0.0
    ninstance = 0.0
    nloss = 0.0
    for (x,y) in dtst
        yhat=pred(w,x)
        for i=1:size(y, 2)
            ncorrect += indmax(y[:,i]) == indmax(yhat[:, i]) ? 1.0 : 0.0
            ninstance+=1
        end
        nloss+= loss(w,x,y)
    end
    return (1-(ncorrect/ninstance), nloss/ninstance)
end

#   MAIN LOOP

# Hyperparameters
your_seed = 1;
EPOCHS    = 100;
BATCHSIZE = 100;
LR        = 0.15;
h   = ((28,28,1), (5,5,3), 10);

srand(your_seed)

dtrn=minibatch(xtrn, ytrn, BATCHSIZE; xtype=KnetArray{Float32},ytype=KnetArray{Float32})
dtst=minibatch(xtst, ytst, BATCHSIZE; xtype=KnetArray{Float32}, ytype=KnetArray{Float32})#returns list of tuples of (x,y)


weights = initweights(h);

println(accuracy(weights,dtrn)[1])
println(accuracy(weights,dtrn)[2])
println(accuracy(weights,dtst)[1])
println(accuracy(weights,dtrn)[2])


@time for epoch=1:EPOCHS # @time helps you to have an idea about your convergence time
    train(weights, dtrn, LR)
    println(accuracy(weights,dtrn)[1])
    println(accuracy(weights,dtrn)[2])
    println(accuracy(weights,dtst)[1])
    println(accuracy(weights,dtrn)[2])
end
