------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'cudnn'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "gen_grasps/")      subdirectory to save logs
  -n,--network       (default "logs512_grasp/adversarial.net")          reload pretrained network
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  --scale            (default 64)          scale of images to train on
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

print('<trainer> reloading previously trained network: ' .. opt.network)
tmp = torch.load(opt.network)
model_D = tmp.D
model_G = tmp.G

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
end

-- Get examples to plot
function getSamples(N)
  local numperclass = numperclass or 10
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim)

  -- Generate samples
  noise_inputs:normal(0, 1)
  local samples = model_G:forward(noise_inputs)
  samples = nn.HardTanh():forward(samples)
  local to_plot = {}
  for i=1,N do
    to_plot[#to_plot+1] = samples[i]:float()
  end

  return to_plot
end

for frame=1,1000 do
  print(frame)
  local to_plot = getSamples(100)
  torch.setdefaulttensortype('torch.FloatTensor')
  local formatted = image.toDisplayTensor({input=to_plot, nrow=10})
  formatted:float()
  image.save(opt.save..frame..'.png', formatted)
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
end