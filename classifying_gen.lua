require 'torch'
require 'nn'

require 'optim'

local model_utils=require 'util.model_utils'
require 'image'
require 'gnuplot'
opt={
    ngen=3,
    ndata=100000,             -- number of batches per epoch
    ncentres=6,
    std_dev=0.1,
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    nz=3,
    batchSize=32,
    ndim=2,
    R=5,
    nvis=3,                    -- Number of samples to be visualized
    save_freq=1,
    exp_name='3gen/',
    niter=1200
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
local real_label=ngen+1
local fake_labels=torch.linspace(1,ngen,ngen)



local G={}

G.netG1= nn.Sequential()
G.netG1:add(nn.Linear(3,128))
G.netG1:add(nn.ReLU())
G.netG1:add(nn.Linear(128,128))
G.netG1:add(nn.ReLU())
G.netG1:add(nn.Linear(128,ndim))


for i=2,ngen do
    G['netG'..i]=G.netG1:clone()
end

local netD=nn.Sequential()
netD:add(nn.Linear(ndim,128))
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
        real=input:normal(0,std_dev)
        for j=1,opt.batchSize do
            k=randints[j]
            --print('k '..k)
            --print('x: value '..tostring(R*math.cos((2*k*math.pi)/ncentres)))
            --print('y: value '..tostring(R*math.sin((2*k*math.pi)/ncentres)))
            real[j][1]=torch.normal(0,std_dev)+R*math.cos((2*k*math.pi)/ncentres)
            real[j][2]=torch.normal(0,std_dev)+R*math.sin((2*k*math.pi)/ncentres)
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
    local randints=torch.Tensor(opt.batchSize):random(1,ncentres)
    for j=1,opt.batchSize do
        k=randints[j]
        real[j][1]=torch.normal(0,std_dev)+R*math.cos((2*k*math.pi)/ncentres)
        real[j][2]=torch.normal(0,std_dev)+R*math.sin((2*k*math.pi)/ncentres)
    end
    if epoch%save_freq==0 then
        paths.mkdir(opt.exp_name..tostring(epoch))
        file=io.open('gaussians.txt','w')
        for i=1,ngen do
            for j=1,nvis do
                noise=noise:normal(0,1)
                local fake=G['netG'..i]:forward(noise)
                vis[{ { 1+(j-1)*opt.batchSize,j*opt.batchSize},{1,ndim}}]=fake
           end
            io.output(file)
            for j=1,nvis*opt.batchSize do
                io.write(string.format('%d %f %f\n',i,vis[j][1],vis[j][2]))
            end
        end
        io.close(file)
        gnuplot.pngfigure(opt.exp_name ..tostring(epoch)..'/out.png' )
        gnuplot.raw("plot 'gaussians.txt' using 2:3:(sprintf('%d', $1)) with labels point pt 7 offset char 0.5,0.5 notitle")
        gnuplot.grid(true)
        --gnuplot.scatter3(torch.zeros(nvis*opt.batchSize)  , vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}]  )
        --gnuplot.scatter3( vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}] , torch.zeros(nvis*opt.batchSize)  )
        gnuplot.plotflush()
        gnuplot.close()
        
        inp_file=io.open('input.txt','w')
        io.output(inp_file)
        for k=1,opt.batchSize do
            io.write(string.format('%d %f %f\n',0,real[k][1],real[k][2]))
        end
        io.close(inp_file)
        gnuplot.pngfigure(opt.exp_name ..tostring(epoch)..'/input.png' )
        gnuplot.raw("plot 'input.txt' using 2:3:(sprintf('%d', $1)) with labels point pt 7 offset char 0.5,0.5 notitle")
        gnuplot.grid(true)
        --gnuplot.scatter3(torch.zeros(nvis*opt.batchSize)  , vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}]  )
        --gnuplot.scatter3( vis[{{1,nvis*opt.batchSize},1}] ,  vis[{{1,nvis*opt.batchSize},2}] , torch.zeros(nvis*opt.batchSize)  )
        gnuplot.plotflush()
        gnuplot.close() 
    end
end
