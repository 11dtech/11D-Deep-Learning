--[[
BASIC PROGRAM INFORAMTION:
Author: Lake Chen
Script Description: this is a hugely simplified feedforward network inspired by human visual cortex strcture. This program implements 6 abstract layers of neural networks, and each layer employs one to several layer of neurons. The 6 abstract layers of networks are call V1-V6. Below is the connection relationship between the layers:

[Layer name]	[Input layer]		[Output layer]
[V1]				[visible]		 	[V2, V5]
[V2]				[V1]			 		[V3, V4, V5]
[V3]				[V2]			 		[V5]	
[V4]				[V2]			 		[V5]
[V5]				[V1, V2, V3, V4]	[V6]
[V6]				[V5]					[output]

Script Testing Command:
>>itorch
>>dofile('gitTorch/Deep01/Deep01.lua')
[image output must be tested using itorch notebook]

]]


require 'nn'
require 'dp'
require 'rnn'
require 'torch'
require 'paths'
require 'optim'


--[[Layer size parameters]]
V1out = 80
	V2in = V1out
V2out = 60
	V3in = V2out
V3out = 30
	V4in = V2out
V4out = 30
	V5in = V1out + V2out + V3out + V4out
V5out = 40
	V6in = V5out
V6out = 10

--[[Layer Definition]]
V1 = nn.Sequential() --Primary visual cortex
V1_1 = nn.SpatialConvolution(3, 6, 5, 5) -- 3 input image channel, 6 output channel, 5x5 convolution kernel
V1_2 = nn.ReLU()	-- non-linearity
V1_3 = nn.SpatialMaxPooling(2, 2, 2, 2)	-- A max-pooling operation that looks at 2x2 windows and finds the max.
V1_4 = nn.SpatialConvolution(6, 16, 5, 5)
V1_5 = nn.ReLU()
V1_6 = nn.SpatialMaxPooling(2, 2, 2, 2)	-- A max-pooling operation that looks at 2x2 windows and finds the max.
V1_7 = nn.View(16*5*5)	--reshapes from 3D tensor of 16x5x5 into 1D tensor of 16*5*5
V1_8 = nn.Linear(16*5*5, V1out)	-- Linearize the V1 output to size 250
V1_9 = nn.ReLU()
V1:add(V1_1)
V1:add(V1_2)
V1:add(V1_3)
V1:add(V1_4)
V1:add(V1_5)
V1:add(V1_6)
V1:add(V1_7)
V1:add(V1_8)
V1:add(V1_9)


V2 = nn.Sequential()	--Secondary visual cortex(prestrate cortex);
V2_1 = nn.Linear(V2in, V2out)
V2_2 = nn.ReLU()
V2:add(V2_1)
V2:add(V2_2)


V3 = nn.Sequential()	--V3 cortext layer
V3_1 = nn.Linear(V3in,V3out)
V3_2 = nn.ReLU()
V3:add(V3_1)
V3:add(V3_2)


V4 = nn.Sequential()	--V4 cortex layer
V4_1 = nn.Linear(V4in,V4out)
V4_2 = nn.ReLU()
V4:add(V4_1)
V4:add(V4_2)


V5 = nn.Sequential()	--middle temporal, V5 layer
V5_1 = nn.Linear(V5in,V5out)
V5_2 = nn.ReLU()
V5:add(V5_1)
V5:add(V5_2)


V6 = nn.Sequential()	--V6 layer, final layer
V6_1 = nn.Linear(V6in,V6out)
V6:add(V6_1)


--[[Substructure Definition]]
V12 = nn.Sequential()	--V1 and V2 layers are in sequential order
V12:add(V1)
V12:add(V2)


V1234 = nn.Sequential()
	V34 = nn.Concat(1)	--V3 and V4 layers both take inputs from V2 layer and the results are concatenated
	V34:add(V3)
	V34:add(V4)
V1234:add(V12)
V1234:add(V34)


--[[Global Structure Definition]]
net = nn.Sequential()	-- Summarizing the entire structure
	V5in = nn.Concat(1)	--Three concatenated paths
	V5in:add(V1)	--	*	-->120
	V5in:add(V12)	--	*-->120-->80
	V5in:add(V1234)	--	*-->120-->80-->60[2 paths]
net:add(V5in)	-- > Input size: (120+80+60*2)
net:add(nn.ReLU())	--Rectify the concatenated ouputs
net:add(V5)	--320-->40
net:add(V6)	--40-->10
net:add(nn.LogSoftMax())	-- converts the output to a log-probability. Useful for classification problems


--[[Print Out Netowrk Structure]]
--print('Network Structure: \n' .. net:__tostring());	--Print the network


--[[Load Data]]
if (not paths.filep("cifar10torchsmall.zip")) then
	os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
	os.execute('unzip cifar10torchsmall.zip')
end

trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer',
			'dog', 'frog', 'horse', 'ship', 'truck'}
--print(trainset)

--[[Index Data]]
setmetatable(trainset,
	{__index = function(t, i)
				  		return {t.data[i], t.label[i]}
				  end}
)
trainset.data = trainset.data:double()	--convert data from a ByteTensor to a DoubleTensor

function trainset:size()
	return self.data:size(1)
end


--[[Normalize Data]]
mean = {}	--store the mean, to normalize the test set
stdv = {}	--store the standard-deviation
for i=1,3 do
	--[{image index},{channel},{vertical pixel},{horizontal pixel}]
	mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()	--mean of each channel
	trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])	--zero out the mean
	
	stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()	--std for each channel
	trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])	--scale for std
end


--[[Define Loss Function]]
criterion = nn.ClassNLLCriterion()	-- a negative log-likelihood criterion for multi-class classification


--[[Train the Network]]
optim_params = { learningRate = 0.001, momentum = 0.5, coefL1=0, coefL2=0.001, maxIteration=10}

-- retrieve parameters and gradients
parameters,gradParameters = net:getParameters()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat('/home/lakechen/gitTorch/Deep01', 'train.log'))
testLogger = optim.Logger(paths.concat('/home/lakechen/gitTorch/Deep01', 'test.log'))

-- get training data size
sampleSize = (#trainset.data)[1]

-- get total training data size
trainSize = (optim_params.maxIteration)*sampleSize

for epoch = 1,optim_params.maxIteration do

	 -- initialize time
   local time = sys.clock()

	for i = 1, sampleSize do	--Iterate through the entire set

		--display the progress
		xlua.progress((epoch-1)*sampleSize+i,trainSize)

		--define inputs and labels
		input = trainset.data[i]
		target = trainset.label[i]

		local func = function(x)

			-- get new parameters
			if x ~= parameters then
	            parameters:copy(x)
	      end
	
			-- reset gradients
	      gradParameters:zero()
	
			-- evaluate function
	      local output = net:forward(input)
	      local f = criterion:forward(output, target)
			
			 -- estimate df/dW
	       local df_do = criterion:backward(output, target)
	       net:backward(input, df_do)
			
			-- penalties (L1 and L2):
	      if optim_params.coefL1 ~= 0 or optim_params.coefL2 ~= 0 then
	         -- locals:
	         local norm,sign= torch.norm,torch.sign
	
	         -- Loss:
	         f = f + optim_params.coefL1 * norm(parameters,1)
	         f = f + optim_params.coefL2 * norm(parameters,2)^2/2
	
	         -- gradients:
	         gradParameters:add( sign(parameters):mul(optim_params.coefL1) + parameters:clone():mul(optim_params.coefL2) )
	      end

	      -- update confusion
   	   confusion:add(output, target)
		
			return f,gradParameters
		end
		optim.sgd(func, parameters, optim_params)
	end

	--Output confusion matrix
	print(confusion)
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

	-- time taken
   time = sys.clock() - time
   time = time / sampleSize
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
	
	-- save/log current net
   local filename = paths.concat('/home/lakechen/gitTorch/Deep01', 'trained.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
	torch.save(filename, net)

	--next epoch
	epoch = epoch + 1
end

trainLogger:style{['% mean class accuracy (train set)'] = '-'}
--testLogger:style{['% mean class accuracy (test set)'] = '-'}
trainLogger:plot()
--testLogger:plot()

--[[Native Training Method]]
--trainer = nn.StochasticGradient(net, criterion)
--trainer.learningRate = 0.001
--trainer.maxIteration = 50	--number of epochs of training
--trainer:train(trainset)


--[[Test Network Accuracy]]
--[[Convert to Double Tensor and Normalize Test Data W/ Mean and STD of Training Data]]
testset.data = testset.data:double()
for i=1,3 do
	testset.data[{ {}, {i}, {}, {}}]:add(-mean[i])
	testset.data[{ {}, {i}, {}, {}}]:div(stdv[i])
end


--[[Test Network Accuracy]]
correct = 0
for i=1,10000 do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
   local confidences, indices = torch.sort(prediction, true)
	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
print('Network accuracy: ', 100*correct/10000, ' %')
