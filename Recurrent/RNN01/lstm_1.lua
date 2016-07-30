require 'rnn'
require 'gnuplot'
require 'optim'

-- hyper-parameters 
batchSize = 1
rho = 30 -- sequence length
inputSize = 1
outputSize = 10
decayPeriod = 30
maxIteration = 200
lead = 1	--prediction steps ahead
optim_params = { learningRate = 0.1, momentum = 0.5, coefL1=0, coefL2=0}

-- build simple recurrent neural network
local rnn = nn.Sequential()
	:add(nn.LSTM(inputSize, outputSize, rho))
	:add(nn.Linear(outputSize, inputSize))
   :add(nn.Sigmoid())
rnn = nn.Recursor(rnn, rho)

print(rnn)
parameters,gradParameters = rnn:getParameters() -- retrieve parameters and gradients
-- build criterion
criterion = nn.MSECriterion()

-- load signal data
signal = torch.load('gitTorch/Recurrent/Data/Signal.t7')
local signal_min = signal[{ {}, {2} }]:min()
local signal_max = signal[{ {}, {2} }]:max()
local signal_shift = signal[{ {}, {2} }]:add(-signal_min)
local signal_range = signal_max-signal_min
signal[{ {}, {2} }] = signal_shift:div(signal_range)
-- load trend data
trend = torch.load('gitTorch/Recurrent/Data/Trend.t7')
local trend_shift = trend[{ {}, {2} }]:add(-signal_min)
trend[{ {}, {2} }] = trend_shift:div(signal_range)
-- load test data
test = torch.load('gitTorch/Recurrent/Data/Test.t7')
local test_shift = test[{ {}, {2} }]:add(-signal_min)
test[{ {}, {2} }] = test_shift:div(signal_range)
-- load kalman prediction
kalman_series = torch.load('gitTorch/Recurrent/Data/Kalman.t7')
local kalman_shift = kalman_series:add(-signal_min)
kalman_series = kalman_shift:div(signal_range)

kalman = torch.Tensor(#signal)
kalman[{ {}, {1} }] = signal[{ {}, {1} }]
kalman[{ {}, {2} }] = kalman_series

prediction = torch.Tensor(#signal):fill(0.5)
prediction[{ {}, {1} }] = signal[{ {}, {1} }]
progress_train = torch.Tensor(maxIteration):fill(0)
progress_test = torch.Tensor(maxIteration):fill(0)
progress_true = torch.Tensor(maxIteration):fill(0)

--training
local index = 1
local iteration = 1
while iteration <= maxIteration do
	local train_err,test_err,true_err = 0,0,0
	for index = 1,((#signal)[1]-rho+1-lead) do --skip the prediciton after the last point
		outputs = {}
		local f = 0  --NEW
		local err = 0
		rnn:zeroGradParameters()
			gradParameters:zero()  -- reset gradients
   	rnn:forget() --forget all past time-steps
	
		--forward
		for step = 1, rho do
			outputs[step] = rnn:forward(signal[{ {index+step-1}, {2} }])
			f = criterion:forward(outputs[step], signal[{ {index+step}, {2} }])  --NEW
		end
		
		prediction[{ {index+rho-1+lead}, {} }] = torch.cat(torch.DoubleTensor({index+rho-1+lead}), torch.DoubleTensor(outputs[rho]))  --last input before making prediction: index+rho-1
		train_err = train_err + criterion:forward(outputs[rho], signal[{ {index+rho-1+lead}, {2} }])
		test_err = test_err + criterion:forward(outputs[rho], test[{ {index+rho-1+lead}, {2} }])
		true_err = true_err + criterion:forward(outputs[rho], trend[{ {index+rho-1+lead}, {2} }])
		
		local gradOutputs, gradInputs = {}, {}
   	for step=rho,1,-1 do -- reverse order of forward calls
     		gradOutputs[step] = criterion:backward(outputs[step], signal[{ {index+step-1+lead}, {2} }])
      	gradInputs[step] = rnn:backward(signal[{ {index+step-1}, {2} }], gradOutputs[step])
   	end
		
		if optim_params.coefL1 ~= 0 or optim_params.coefL2 ~= 0 then  -- penalties (L1 and L2)
      	local norm,sign= torch.norm,torch.sign  -- locals
      	f = f + optim_params.coefL1 * norm(parameters,1)  -- L1 Loss
      	f = f + optim_params.coefL2 * norm(parameters,2)^2/2  -- L2 Loss     
	   	gradParameters:add( sign(parameters):mul(optim_params.coefL1) + parameters:clone():mul(optim_params.coefL2))  -- gradients
   	end

		local function func(x)  --dummie function
			return f, gradParameters
		end
		optim.sgd(func, parameters, optim_params)
		--rnn:updateParameters(lr)  -- update
	end

	AE_train = math.sqrt(train_err/((#signal)[1]-rho))
	AE_test = math.sqrt(test_err/((#signal)[1]-rho))
	AE_true = math.sqrt(true_err/((#signal)[1]-rho))

	progress_train[{iteration}] = AE_train  --average prediction error
	progress_test[{iteration}] = AE_test
	progress_true[{iteration}] = AE_true
	local diff = AE_train - AE_true
	print(string.format("Iteration %d ; Mean MSE Training Err = %f; Mean MSE True Err = %f; Err Diff = %f", iteration, AE_train, AE_true, diff ))
	
	if iteration > 1 then
	gnuplot.figure(1)	
		gnuplot.plot({'Signal', signal[{ {1,500}, {} }], '-'}, {'Trend', trend[{ {1,500}, {} }], '~'}, {'RNN', prediction[{ {1,500}, {} }], 'with points ps 0.75'}, {'Kalman', kalman[{ {1,500} }], 'with points ps 0.75'})
			gnuplot.xlabel('Timestep')
			gnuplot.ylabel('Normalized Index')
			gnuplot.title('Training Results')
			gnuplot.grid(true)
	end
	gnuplot.figure(2)	
		gnuplot.plot({'Train Error',progress_train,'-'}, {'Test Error',progress_test,'-'}, {'True Error', progress_true,'-'})
			gnuplot.xlabel('Timestep')
			gnuplot.ylabel('Average Prediction Error')
			gnuplot.title('Training Progress')
			gnuplot.grid(true)
		
	if iteration%decayPeriod==0 then
		optim_params.learningRate = optim_params.learningRate/2
		print('Learning Rate: ', optim_params.learningRate)
	end

	iteration = iteration + 1
end
