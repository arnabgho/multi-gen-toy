require 'torch'
require 'nn'
require 'optim'
require 'pl'

opt={
    folder='datasets',
    data_name='test',
    num_samples=768000,
    filename='specs.txt',
    out_name='input.txt',
    t7_filename='data.t7',
    h5_filename='data.h5',
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)



require 'hdf5'
local data_file=hdf5.open(paths.concat(opt.folder,opt.data_name,opt.h5_filename),'r')
local data=data_file:read('data'):all()
data_file:close()
print(paths.concat(opt.folder,opt.data_name,opt.t7_filename))
torch.save(paths.concat(opt.folder,opt.data_name,opt.t7_filename),data)


