package n2

import (
	"github.com/e74000/alg"
	"gonum.org/v1/gonum/mat"
)

var (
	tanh     = alg.Tanh{X: alg.X{}}
	sigmoid  = alg.Div{N: alg.S(1), D: alg.Add{A: alg.S(1), B: alg.Exp{X: alg.Sx{S: -1}}}}
	softPlus = alg.Ln{X: alg.Add{A: alg.S(1), B: alg.Exp{X: alg.X{}}}}
	silu     = alg.Div{N: alg.X{}, D: alg.Add{A: alg.S(1), B: alg.Exp{X: alg.Sx{S: -1}}}}
)

func NewActTanH() *ActivationLayer {
	return NewActivation(tanh)
}

func NewActSigmoid() *ActivationLayer {
	return NewActivation(sigmoid)
}

func NewActSoftPlus() *ActivationLayer {
	return NewActivation(softPlus)
}

func NewActSiLU() *ActivationLayer {
	return NewActivation(silu)
}

func NewActivation(aFunc alg.Term) (l *ActivationLayer) {
	l = &ActivationLayer{
		AFunc: aFunc,
		PFunc: aFunc.Dx(),
	}

	return l
}

type ActivationLayer struct {
	AFunc      alg.Term
	PFunc      alg.Term
	inputCache []*mat.Dense
}

func (l *ActivationLayer) Forward(inputs []*mat.Dense) []*mat.Dense {
	out := make([]*mat.Dense, len(inputs))
	for d := 0; d < len(inputs); d++ {
		r, c := inputs[d].Dims()
		out[d] = mat.NewDense(r, c, nil)
		out[d].Apply(wrap(l.AFunc.E), inputs[d])
	}

	copyT3(&l.inputCache, &inputs)

	return out
}

func (l *ActivationLayer) Backward(outputGradient []*mat.Dense, _ float64) (inputGradient []*mat.Dense) {
	inputGradient = make([]*mat.Dense, len(outputGradient))

	for d := 0; d < len(outputGradient); d++ {
		r, c := l.inputCache[d].Dims()
		inputGradient[d] = mat.NewDense(r, c, nil)

		inputGradient[d].Apply(wrap(l.PFunc.E), l.inputCache[d])

		inputGradient[d].MulElem(outputGradient[d], inputGradient[d])
	}

	return inputGradient
}
