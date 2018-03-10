using Knet
using Compat,GZip # Helpers to read the MNIST (Like lab-2)

# We need to define preprocessing functions,e.g. downloading and loading the data
function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

# That function is going to load raw data to the ram so that you will be able to use it
function loaddata()
    info("Loading MNIST...")
    xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
    xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
    ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
    ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
    return (xtrn, ytrn, xtst, ytst)
end

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


function vnet(w,x,; pdrop=(0,0.2,0.2))
    for i=1:2:length(w)
        if ndims(w[i])==4 #it means convolution layer
            x = dropout(x, pdrop[i==1?1:2])
            x=conv4(w[i],x) .+ w[i+1]
            x=pool(relu.(x))
        elseif ndims(w[i])==2 #it means fully connected layer
            x = dropout(x, pdrop[i==1?1:3])
            x=w[i]*mat(x) .+ w[i+1]
            if i < length(w)-1
                x = relu.(x)
            end
        end
    end
    return x
end

function loss(w,x,ygold)
    ypred=vnet(w,x)
    prob=logp(ypred,[1])
    J=-sum(ygold.*prob)
    return J
end

lossgradient=grad(loss)

function train(w, dtrn, lr)
    p=optimizers(w,Adadelta)
        for (x,y) in dtrn
            g=lossgradient(w,x,y)
            update!(w,g,p)
        end
    return w

end

function accuracy(weights, dtst, pred=vnet)
    w = weights
    ncorrect = 0
    ninstance = 0
    nloss = 0
    for (x,y) in dtst
        ypred=pred(w,x)
        ypred[ypred.==maximum(ypred,1)] = 1
        ypred[ypred.!=1] = 0
        ncorrect+=sum(ypred.*y)
        nloss+= loss(w,x,y)
        ninstance += size(y,2)
    end
    return ((ncorrect/ninstance), nloss/ninstance)
end

#   MAIN LOOP

# Hyperparameters
your_seed = 1;
EPOCHS    = 200;
BATCHSIZE = 100;
LR        = 0.01;
h   = ((28,28,1), (5,5,8),(3,3,8), 10);

srand(your_seed)

# ========== DATA RELATED ============
xtrnraw, ytrnraw, xtstraw, ytstraw = loaddata(); # These are raw data and not suitable to direct usage
# Following line takes the xtrnraw -> normalize, reshape and converts to Float32 data type
xtrn = convert(Array{Float32}, reshape(xtrnraw ./ 255, 28, 28, 1, div(length(xtrnraw), 784)));
xtst = convert(Array{Float32}, reshape(xtstraw ./ 255, 28, 28, 1, div(length(xtstraw), 784)));

# We have 10 class classification 0 is represented as 10th class
ytrnraw[ytrnraw.==0]=10; # That would be useful to understand that line
ytrn = convert(Array{Float32}, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)));
ytstraw[ytstraw.==0]=10;
ytst = convert(Array{Float32}, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)));

dtrn=minibatch(xtrn, ytrn, BATCHSIZE; xtype=KnetArray{Float32},ytype=KnetArray{Float32})
dtst=minibatch(xtst, ytst, BATCHSIZE; xtype=KnetArray{Float32}, ytype=KnetArray{Float32})#returns list of tuples of (x,y)



weights = initweights(h);

#a=accuracy(weights,dtrn)
#b=accuracy(weights,dtst)
#println(a[1])
#println(a[2])
#println(b[1])
#println(b[2])
report(epoch)=println(:epoch,epoch,:trn,accuracy(weights,dtrn),:tst,accuracy(weights,dtst))
report(0)


@time for epoch=1:EPOCHS # @time helps you to have an idea about your convergence time
    train(weights, dtrn, LR)
    report(epoch)
    ###a=accuracy(weights,dtrn)
    ##b=accuracy(weights,dtst)
    #println(a[1])
    #println(a[2])
    #println(b[1])
    #println(b[2])
end
