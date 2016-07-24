require 'gnuplot'

function sine(t,amp, period, phase)
	local value = amp*math.sin(phase/180*math.pi+t*2*math.pi/period);
	return value
end

function dc(shift)
	return shift
end

function gaussian()
    return math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) / 2
end

dt = 1
period = 50
amplitude = 1
seriesLength = 10*period

data = torch.Tensor(seriesLength,2)
trend = torch.Tensor(seriesLength,2)

for t=1,seriesLength do
	local func = sine(t, 0.5, 50, 0) + sine(t, 0.3, 100, 0) + sine(t, 0.4, 125, 45) + sine(t, 0.5, 125, 75)
	local value = func + 0.4*gaussian()

	data[{ {t} }] = torch.Tensor({{t, value}})
	trend[{ {t} }] = torch.Tensor({{t, func}})
end

--Kalman Filter
kalman = torch.Tensor(seriesLength):fill(0.5)  --index prediction
prediction = torch.Tensor(seriesLength):fill(0.5)
gain = torch.Tensor(seriesLength):fill(0.5)
perr = torch.Tensor(seriesLength):fill(0.5)
r = 0.4
a = 1.04

for k=2, seriesLength do
	--prediction phase
	kalman[k] = a*kalman[k-1]
	prediction[k] = kalman[k]
	perr[k] = a*perr[k-1]*a
	--update phase
	gain[k] = perr[k]/(perr[k]+r)
	kalman[k] = kalman[k] + gain[k]*(data[k][2]-kalman[k])
	perr[k] = (1-gain[k])*perr[k]
end

torch.save('gitTorch/Recurrent/Data/Signal.t7', data)
torch.save('gitTorch/Recurrent/Data/Trend.t7', trend)
torch.save('gitTorch/Recurrent/Data/Kalman.t7', prediction)

gnuplot.plot({data}, {trend}, {prediction})
