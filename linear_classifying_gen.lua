require 'torch'
require 'nn'
require 'optim'

local model_utils=require 'util.model_utils'
require 'image'
require 'gnuplot'
opt={
    ncircles=3,
    distC=100,
    ngen=3,
    nz=3,
    batchSize=256,
    R=5,
    ncentres=6,
    ndata=100000,             -- number of batches per epoch
    std_dev=0.1,
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    ndim=1,
    nvis=3,                    -- Number of samples to be visualized
    save_freq=1,
    exp_name='linearGen',
    niter=1200,
    batchnorm=true,
    nbin=20,
    batchnormD=false,
}

G={}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

opt.manualSeed = torch.random(1, 10000) -- fix seed
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
local real_label=ngen+1
local fake_labels=torch.linspace(1,ngen,ngen)



local G={}

G.netG1= nn.Sequential()
G.netG1:add(nn.Linear(3,128))
if opt.batchnorm==true then
    G.netG1:add(nn.BatchNormalization(128))    
end    
G.netG1:add(nn.ReLU())
G.netG1:add(nn.Linear(128,128))
if opt.batchnorm==true then
    G.netG1:add(nn.BatchNormalization(128))    
end
G.netG1:add(nn.ReLU())
G.netG1:add(nn.Linear(128,ndim))

for i=2,ngen do
    G['netG'..i]=G.netG1:clone()
end

local netD=nn.Sequential()
netD:add(nn.Linear(ndim,128))
if opt.batchnormD==true then
    netD:add(nn.BatchNormalization(128))
end
netD:add(nn.ReLU())
netD:add(nn.Linear(128,ngen+1))

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

local errG=0
local errD=0
local fDx=function(x)
    gradParametersD:zero()
    for i=1,ngen do
        input:normal(0,std_dev)
        local randints=torch.Tensor(opt.batchSize):random(1,ncentres)
        --local randshifts=torch.Tensor(opt.batchSize):random(1,ncircles)
        --real=input:normal(0,std_dev)
        for j=1,opt.batchSize do
            k=randints[j]
            if ncentres%2==1 then
                k=k-(ncentres+1)/2
            else
               if k%2==0 then
                   k=-(k-1)
               else
                   k=k
               end
            end
            real[j][1]=torch.normal(0,std_dev)+k*R   -- +R*math.cos((2*k*math.pi)/ncentres)
            --real[j][2]=torch.normal(0,std_dev)+k    -- R*math.sin((2*k*math.pi)/ncentres)
            --k=randshifts[j]
            --real[j][1]=real[j][1]+distC*math.cos((2*k*math.pi)/ncircles)
            --real[j][2]=real[j][2]+distC*math.sin((2*k*math.pi)/ncircles)
        end
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
    for iter=1,ndata/opt.batchSize do
        optim.adam( fDx, parametersD ,optimStateD)
        optim.adam( fGx, parametersG ,optimStateG)
        --print('epoch '..epoch..' iter '..iter.. ' errG '..tostring(errG)..' errD '..tostring(errD))
    end
    --local randints=torch.Tensor(opt.batchSize):random(1,ncentres)
    --local randshifts=torch.Tensor(opt.batchSize):random(1,ncircles)
    --for j=1,opt.batchSize do
    --    k=randints[j]
    --    real[j][1]=torch.normal(0,std_dev)+R*math.cos((2*k*math.pi)/ncentres)
    --    real[j][2]=torch.normal(0,std_dev)+R*math.sin((2*k*math.pi)/ncentres)
    --    k=randshifts[j]
    --    if k==1 then
    --        if torch.rand(1)[1]>0.1 then 
    --            k=2
    --        end
    --    end
    --    real[j][1]=real[j][1]+distC*math.cos((2*k*math.pi)/ncircles)
    --    real[j][2]=real[j][2]+distC*math.sin((2*k*math.pi)/ncircles)
    --end
    if epoch%save_freq==0 then
        local dir = opt.exp_name..'_'..tostring(ncircles)..'_'..tostring(distC)..'_'..tostring(ngen)..'_'..tostring(nz)..'_'..tostring(batchSize)..'_'..tostring(R)..'_'..tostring(ncentres)..'/'..tostring(epoch)
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
            gnuplot.hist(vis,nbin)
            gnuplot.plotflush()
            gnuplot.close()
            for j=1,nvis*opt.batchSize do
                io.write(string.format('%d %f\n',i,vis[j][1]))
            end
        end
        io.close(file)
        --gnuplot.pngfigure(dir..'/out.png' )
        --gnuplot.raw("plot '"..dir..'/out.txt'.."' using 2:3:(sprintf('%d', $1)) with labels point pt 7 offset char 0.5,0.5 notitle")
        --gnuplot.grid(true)
        ----gnuplot.scatter3(torch.zeros(nvis*opt.batchSize)  , vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}]  )
        ----gnuplot.scatter3( vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}] , torch.zeros(nvis*opt.batchSize)  )
        --gnuplot.plotflush()
        --gnuplot.close()
        --
        inp_file=io.open(dir..'/input.txt','w')
        io.output(inp_file)
        for k=1,opt.batchSize do
            io.write(string.format('%d %f\n',0,real[k][1]))
        end
        io.close(inp_file)
        gnuplot.pngfigure(dir..'/input.png' )
        gnuplot.hist(real,ngen*nbin)
        gnuplot.plotflush()
        gnuplot.close()
        --gnuplot.raw("plot '"..dir..'/input.txt'.."' using 2:3:(sprintf('%d', $1)) with labels point pt 7 offset char 0.5,0.5 notitle")
        --gnuplot.grid(true)
        ----gnuplot.scatter3(torch.zeros(nvis*opt.batchSize)  , vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}]  )
        ----gnuplot.scatter3( vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}] , torch.zeros(nvis*opt.batchSize)  )
        --gnuplot.plotflush()
        --gnuplot.close()
    end
end
