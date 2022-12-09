package n2

import (
	"gonum.org/v1/gonum/mat"
)

func tile(vec *mat.Dense, n int) *mat.Dense {
	r, _ := vec.Dims()

	out := mat.NewDense(r, n, nil)

	out.Apply(func(i, _ int, _ float64) float64 {
		return vec.At(i, 0)
	}, out)

	return out
}

func identity(n int) *mat.Dense {
	m := mat.NewDense(n, n, nil)

	m.Apply(func(i, j int, _ float64) float64 {
		if i == j {
			return 1
		}
		return 0
	}, m)

	return m
}

// SoftmaxLayer contains unused exported parameter Ignore as a hack to help with gob encoding
type SoftmaxLayer struct {
	Ignore      byte
	outputCache []*mat.Dense
}

func NewSoftmaxLayer() (l *SoftmaxLayer) {
	return new(SoftmaxLayer)
}

func (l *SoftmaxLayer) Forward(inputs []*mat.Dense) []*mat.Dense {
	copyT3(&l.outputCache, &inputs)

	sum := mat.Sum(inputs[0])

	l.outputCache[0].Apply(func(_, _ int, v float64) float64 {
		return v / sum
	}, inputs[0])

	return l.outputCache
}

func (l *SoftmaxLayer) Backward(outputGradient []*mat.Dense, _ float64) (inputGradient []*mat.Dense) {
	r, _ := l.outputCache[0].Dims()
	tiled := tile(l.outputCache[0], r)
	ident := identity(r)

	ident.Sub(ident, tiled.T())
	tiled.MulElem(tiled, ident)

	inputGradient = []*mat.Dense{
		mat.NewDense(r, 1, nil),
	}

	inputGradient[0].Product(tiled, outputGradient[0])

	return inputGradient
}
