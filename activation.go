package n2

import (
	"encoding/json"
	"gonum.org/v1/gonum/mat"
	"math"
)

func tanh(v float64) float64 {
	return math.Tanh(v)
}

func tanhPrime(v float64) float64 {
	return 1 - math.Pow(math.Tanh(v), 2)
}

func sigmoid(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func sigmoidPrime(v float64) float64 {
	return (1 / (1 + math.Exp(-v))) * (-math.Exp(-v) / (1 + math.Exp(-v)))
}

func relu(v float64) float64 {
	if v > 0 {
		return v
	}

	return 0
}

func reluPrime(v float64) float64 {
	if v > 0 {
		return 1
	}

	return 0
}

func softPlus(v float64) float64 {
	return math.Log(1 + math.Exp(v))
}

func silu(v float64) float64 {
	return v / (1 + math.Exp(-v))
}

func siluPrime(v float64) float64 {
	return (1 + math.Exp(-v)*(1+v)) / math.Pow(1+math.Exp(-v), 2)
}

func NewActTanH() *ActivationLayer {
	return newActivation(tanh, tanhPrime, "TanH")
}

func NewActSigmoid() *ActivationLayer {
	return newActivation(sigmoid, sigmoidPrime, "Sigmoid")
}

func NewActReLU() *ActivationLayer {
	return newActivation(relu, reluPrime, "ReLU")
}

func NewActSoftPlus() *ActivationLayer {
	return newActivation(softPlus, sigmoid, "ReLU")
}

func NewActSiLU() *ActivationLayer {
	return newActivation(silu, siluPrime, "SiLU")
}

type ActivationLayer struct {
	aFunc func(v float64) float64
	pFunc func(v float64) float64
	aType string

	inputCache []*mat.Dense
}

func newActivation(aFunc, pFunc func(v float64) float64, aType string) (l *ActivationLayer) {
	l = &ActivationLayer{
		aFunc: aFunc,
		pFunc: pFunc,
		aType: aType,
	}

	return l
}

func (l *ActivationLayer) Forward(inputs []*mat.Dense) []*mat.Dense {
	out := make([]*mat.Dense, len(inputs))
	for d := 0; d < len(inputs); d++ {
		r, c := inputs[d].Dims()
		out[d] = mat.NewDense(r, c, nil)
		out[d].Apply(wrap(l.aFunc), inputs[d])
	}

	copyT3(&l.inputCache, &inputs)

	return out
}

func (l *ActivationLayer) Backward(outputGradient []*mat.Dense, _ float64) (inputGradient []*mat.Dense) {
	inputGradient = make([]*mat.Dense, len(outputGradient))

	for d := 0; d < len(outputGradient); d++ {
		r, c := l.inputCache[d].Dims()
		inputGradient[d] = mat.NewDense(r, c, nil)

		inputGradient[d].Apply(wrap(l.pFunc), l.inputCache[d])

		inputGradient[d].MulElem(outputGradient[d], inputGradient[d])
	}

	return inputGradient
}

type ActivationLayerJSON struct {
	LType string
	AType string
}

func (l *ActivationLayer) MarshalJSON() ([]byte, error) {
	lj := &ActivationLayerJSON{
		LType: "Activation",
		AType: l.aType,
	}

	return json.Marshal(lj)
}

func (l *ActivationLayer) UnmarshalJSON(bytes []byte) error {
	lj := &ActivationLayerJSON{}

	var err error
	err = json.Unmarshal(bytes, lj)
	if err != nil {
		return err
	}

	*l = *ATypes[lj.AType]()

	return nil
}
