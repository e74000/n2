package n2

import (
	"gonum.org/v1/gonum/mat"
)

func NewFlatten() (l *FlattenLayer) {
	l = new(FlattenLayer)
	return l
}

type FlattenLayer struct {
	InputShape intV3
}

func (l *FlattenLayer) Forward(input []*mat.Dense) []*mat.Dense {
	r, c := input[0].Dims()
	l.InputShape = intV3{
		X: c,
		Y: r,
		Z: len(input),
	}
	return reshapeT3toT1(input)
}

func (l *FlattenLayer) Backward(outputGradient []*mat.Dense, _ float64) []*mat.Dense {
	return reshapeT1toT3(outputGradient, l.InputShape)
}
