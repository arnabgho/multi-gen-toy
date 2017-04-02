require 'torch'
require 'nn'
require 'optim'

local model_utils=require 'util.model_utils'
require 'image'
require 'gnuplot'
opt={
    ncircles=3,
    distC=100,
    ngen=1,
    nz=1,
    batchSize=256,
    R=5,
    ncentres=6,
    ndata=100000,             -- number of batches per epoch
    std_dev=0.1,
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    ndim=1,
    nvis=1000,                    -- Number of samples to be visualized
    save_freq=50,
    exp_name='headsLinearGen',
    niter=1200,
    batchnorm=true,
    nbin=20,
    batchnormD=false,
    nhid=128,
    folder='datasets',
    data_name='test',
    t7_filename='data.t7',
}


for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

opt.manualSeed =7-- torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

ngen=opt.ngen
ndata=opt.ndata
ncentres=opt.ncentres
std_dev=opt.std_dev
nz=opt.nz
ndim=opt.ndim
R=opt.R
nvis=opt.nvis
save_freq=opt.save_freq
distC=opt.distC
ncircles=opt.ncircles
batchSize=opt.batchSize
nbin=opt.nbin
nhid=opt.nhid
local real_label=ngen+1
local fake_labels=torch.linspace(1,ngen,ngen)



local G={}

netG= nn.Sequential()
netG:add(nn.Linear(nz,nhid))
if opt.batchnorm==true then
    netG:add(nn.BatchNormalization(nhid))    
end    
netG:add(nn.ReLU())


for i=1,ngen do
    G['netG'..i]=nn.Sequential()
    if i==1 then 
        G['netG'..i]:add(netG)
    else
        G['netG'..i]:add(netG:clone('weight','bias','gradWeight','gradBias'))
    end

    G['netG'..i]:add(nn.Linear(nhid,nhid))
    if opt.batchnorm==true then
        G['netG'..i]:add(nn.BatchNormalization(nhid))    
    end
    G['netG'..i]:add(nn.ReLU())
    G['netG'..i]:add(nn.Linear(nhid,ndim))
end

local netD=nn.Sequential()
netD:add(nn.Linear(ndim,nhid))
if opt.batchnormD==true then
    netD:add(nn.BatchNormalization(nhid))
end
netD:add(nn.ReLU())
netD:add(nn.Linear(nhid,ngen+1))

local criterion=nn.CrossEntropyCriterion()
optimStateG = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
optimStateD = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}

local noise=torch.Tensor(opt.batchSize,nz)
local input=torch.Tensor(opt.batchSize,ndim)
local label = torch.Tensor(opt.batchSize)
local noise_cache = torch.Tensor(ngen,opt.batchSize , nz )
local vis=torch.Tensor(nvis*opt.batchSize,ndim)
local real=torch.Tensor(opt.batchSize,ndim)
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = model_utils.combine_all_parameters(G)
local data=torch.load(paths.concat(opt.folder,opt.data_name,opt.t7_filename ))
local errG=0
local errD=0
local fDx=function(x)
    gradParametersD:zero()
    for i=1,ngen do
        input:copy(real)
        --print(input)
        label:fill(real_label)
        local output=netD:forward(input)
        errD=criterion:forward(output,label)
        local df_do=criterion:backward(output,label)
        netD:backward(input,df_do)

        noise:normal(0,1)
        noise_cache[i]=noise
        local fake=G['netG'..i]:forward(noise)
        input:copy(fake)
        label:fill(fake_labels[i])
        local output=netD:forward(input)
        errD=errD+criterion:forward(output,label)
        local df_do=criterion:backward(output,label)
        netD:backward(input,df_do)
    end
    return errD,gradParametersD
end


local fGx=function(x)
    gradParametersG:zero()
    label:fill(real_label)
    errG=0
    for i=1,ngen do
        local output=netD:forward(G['netG'..i].output)
        errG=errG+criterion:forward(output,label)
        local df_do=criterion:backward(output,label)
        local df_dg=netD:updateGradInput(G['netG'..i].output,df_do)
        G['netG'..i]:backward(noise,df_dg)
    end
    return errG,gradParametersG
end


for epoch=1,opt.niter do
    local nbatches=math.floor(data:size(1)/opt.batchSize)
    for iter=1,nbatches do
        real=data[{{1+(iter-1)*opt.batchSize,iter*opt.batchSize  },{1 , ndim}}]
        optim.adam( fDx, parametersD ,optimStateD)
        optim.adam( fGx, parametersG ,optimStateG)
    end
    --print('epoch '..epoch..' errG '..tostring(errG)..' errD '..tostring(errD))
    if epoch%save_freq==0 then
        local name=opt.exp_name .. '_' .. opt.ngen .. '_' .. tostring(opt.batchnorm) .. '_' .. opt.nhid .. '_' .. opt.nz
        local dir = paths.concat(opt.folder,opt.data_name,name ,tostring(epoch))
        paths.mkdir(dir)
        file=io.open(dir..'/out.txt','w')
        for i=1,ngen do
            for j=1,nvis do
                noise=noise:normal(0,1)
                local fake=G['netG'..i]:forward(noise)
                vis[{ { 1+(j-1)*opt.batchSize,j*opt.batchSize},{1,ndim}}]=fake
           end
            io.output(file)
            gnuplot.pngfigure(dir..'/out_'..i..'.png')
            gnuplot.hist(vis) --gnuplot.hist(vis,nbin)
            gnuplot.plotflush()
            gnuplot.close()
            for j=1,nvis*opt.batchSize do
                if ndim==1 then
                    io.write(string.format('%d %f\n',i,vis[j][1]))
                elseif ndim==2 then
                    io.write(string.format('%d %f %f\n',i,vis[j][1],vis[j][2]))
                end
            end
        end
        io.close(file)
    end
end
