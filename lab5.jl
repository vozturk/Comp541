
# Set-Up related files and Hyper-parameters
using Knet
Pkg.installed("ArgParse") == nothing && Pkg.add("ArgParse")
using ArgParse
include(Pkg.dir("Knet","data","wikiner.jl"));

# You are given make_batches function
function make_batches(data, w2i, t2i, batchsize)
    batches = []
    sorted = sort(data, by=length, rev=true)
    for k = 1:batchsize:length(sorted)
        lo = k
        hi = min(k+batchsize-1, length(sorted))
        samples = sorted[lo:hi]
        batch = make_batch(samples, w2i, t2i)
        push!(batches, batch)
    end
    return batches
end

#  You need to implement make_batch function
function make_batch(samples, w2i, t2i)
    input = Int[]
    output = Int[]
    longest = length(samples[1])
    batchsizes = zeros(Int, longest)

    # YOUR ANSWER
     s=length(samples)
    for i in 1:longest
        for j in 1:s
            if length(samples[j])>=i
                push!(input, get(w2i,samples[j][i][1],1))
                push!(output, get(t2i,samples[j][i][2],1))
                batchsizes[i] +=1
            end
        end
    end


    return input, output, batchsizes
end

# w[1]   => weight/bias params for forward LSTM network
# w[2:5] => weight/bias params for MLP+softmax network
# w[6]   => word embeddings
# w[7]   => rnnstruct given by rnninit function
# Hint you mave take a look main function below to better understand its calling convention
function initweights(atype, hidden, words, tags, mlp, embed, usegpu, winit=0.01)
    w = Array{Any}(7)
    # YOUR ANSWER
    we(d...)=atype(randn(d...))
    bi(d...)=atype(zeros(d...))
    w[2]=we(mlp,2*hidden)
    w[3]=bi(mlp,1)
    w[4]=we(tags,mlp)
    w[5]=bi(tags,1)
    w[6]=we(embed,words)
    w[7],w[1]=rnninit(embed,hidden;bidirectional=true,rnnType=:lstm)
    return w
end

function predict(ws, xs, batchsizes)
    # YOUR ANSWER
     x = ws[6][:,xs]
    (y,_) = rnnforw(ws[7],ws[1],x,batchSizes=batchsizes)
    ymlp=relu.(ws[2] * y .+ ws[3])
    y=ws[4]*ymlp .+ ws[5]
    return y
end

# our loss function
function loss(w, x, ygold, batchsizes)
    # YOUR ANSWER
     return nll(predict(w,x,batchsizes),ygold)
end

lossgradient = gradloss(loss) # Knet's automatic gradient calculator function

function train!(w, x, ygold, batchsizes, opt)
    # YOUR ANSWER
     g,lossval=lossgradient(w,x,ygold,batchsizes)
    update!(w,g,opt)
    return lossval
end

function accuracy(w, batches, i2t)
    # YOUR ANSWER
   ncorrect = 0.0
    ntoken = 0.0
    for (x,y,z) in batches
        s=size(y,2)
        ypred=predict(w,x,z)
        #ypred[ypred.==maximum(ypred,1)] = 1
        #ypred[ypred.!=1] = 0
        #ygold=zeros(size(ypred))
        #for i in 1:length(y)
            #ygold[y[i],i]=1
        #end
       # ygold=convert(KnetArray{Float32},ygold)
        #ncorrect+=sum(ypred.*ygold)
        for i=1:size(y, 2)
            b=Array(ypred[:,i])
            ncorrect += indmax(b) == y[i] ? 1.0 : 0.0
        end
        ntoken += s
    end
    tag_acc=ncorrect/ntoken

    return tag_acc
end

# Do not touch this function
function main(args)
    s = ArgParseSettings()
    s.description = "Bidirectional LSTM Tagger in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--embed"; arg_type=Int; default=128; help="word embedding size")
        ("--hidden"; arg_type=Int; default=50; help="LSTM hidden size")
        ("--mlp"; arg_type=Int; default=32; help="MLP size")
        ("--epochs"; arg_type=Int; default=20; help="number of training epochs")
        ("--minoccur"; arg_type=Int; default=6; help="word min occurence limit")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--batchsize"; arg_type=Int; default=100; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}
    #datadir = abspath(joinpath(@__DIR__, "../data/tags"))
    datadir = WIKINER_DIR

    # load WikiNER data
    data = WikiNERData(datadir, o[:minoccur])

    # build model
    nwords, ntags = length(data.w2i), data.ntags
    w = initweights(
        atype, o[:hidden], nwords, ntags, o[:mlp], o[:embed], o[:usegpu])
    opt = optimizers(w, Adam)

    # make batches
    trn = make_batches(data.trn, data.w2i, data.t2i, o[:batchsize])
    dev = make_batches(data.dev, data.w2i, data.t2i, o[:batchsize])

    # train bilstm tagger
    nwords = data.nwords
    println("nwords=$nwords, ntags=$ntags"); flush(STDOUT)
    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_tagged = this_loss = 0
    for epoch = 1:o[:epochs]
        # training
        shuffle!(trn)
        for (k,batch) in enumerate(trn)
            x, ygold, batchsizes = batch
            num_tokens = sum(batchsizes)
            batch_loss = train!(w, x, ygold,  batchsizes, opt)
            this_loss += num_tokens*batch_loss
            this_tagged += num_tokens
        end

        # validation
        dev_start = now()
        tag_acc = accuracy(w, dev, data.i2t)
        dev_time += Int((now()-dev_start).value)*0.001
        train_time = Int((now()-t0).value)*0.001-dev_time

        # report
        @printf("epoch %d finished, loss=%f\n", epoch, this_loss/this_tagged)
        all_tagged += this_tagged
        this_loss = this_tagged = 0
        all_time = Int((now()-t0).value)*0.001
        @printf("tag_acc=%.4f, time=%.4f, word_per_sec=%.4f\n",
                tag_acc, train_time, all_tagged/train_time)
        flush(STDOUT)
    end
end

t00 = now();main("--usegpu")

a=zeros(2,2)

a[1,1]=1
a[1,2]=1

find(a)

a=[0,1,0,0]
